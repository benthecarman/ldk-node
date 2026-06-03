// This file is Copyright its original authors, visible in version control history.
//
// This file is licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. You may not use this file except in
// accordance with one or both of these licenses.

#[path = "../tests/common/mod.rs"]
mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use bitcoin::Amount;
use common::{
	expect_channel_pending_event, expect_channel_ready_event, expect_event,
	generate_blocks_and_wait, premine_and_distribute_funds, random_chain_source, random_config,
	setup_bitcoind_and_electrsd, setup_node, setup_two_nodes_with_store,
};
use criterion::{criterion_group, criterion_main, Criterion};
use electrsd::corepc_node::{Client as BitcoindClient, Node as BitcoinD};
use ldk_node::{Event, Node};
use lightning::ln::channelmanager::PaymentId;
use lightning::routing::router::RouteParametersConfig;
use lightning_invoice::{Bolt11InvoiceDescription, Description};

use crate::common::{open_channel_push_amt, TestChainSource, TestConfig, TestStoreType};

#[derive(Clone, Copy)]
struct StoreBenchConfig {
	name: &'static str,
	store_type: TestStoreType,
}

fn operations_benchmark(c: &mut Criterion) {
	forwarding_benchmark(c);
	channel_open_benchmark(c);
	startup_benchmark(c);
}

fn forwarding_benchmark(c: &mut Criterion) {
	let (bitcoind, electrsd) = setup_bitcoind_and_electrsd();
	let chain_source = TestChainSource::Esplora(&electrsd);
	let runtime =
		tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();

	let mut group = c.benchmark_group("forwarding");
	group.sample_size(10);

	for store_config in store_bench_configs() {
		if !should_register_bench("forwarding", store_config.name) {
			continue;
		}
		let nodes = setup_forwarding_nodes(
			&chain_source,
			&bitcoind,
			&electrsd,
			store_config.store_type,
			&runtime,
		);
		let nodes = Arc::new(nodes);

		group.bench_function(store_config.name, |b| {
			b.to_async(&runtime).iter_custom(|iter| {
				let nodes = Arc::clone(&nodes);

				async move {
					let mut total = Duration::ZERO;
					for _ in 0..iter {
						total += send_forwarded_payments(Arc::clone(&nodes)).await;
					}
					total
				}
			});
		});
	}
}

fn channel_open_benchmark(c: &mut Criterion) {
	let (bitcoind, electrsd) = setup_bitcoind_and_electrsd();
	let chain_source = random_chain_source(&bitcoind, &electrsd);
	let runtime =
		tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();

	let mut group = c.benchmark_group("channel_open");
	group.sample_size(10);

	for store_config in store_bench_configs() {
		if !should_register_bench("channel_open", store_config.name) {
			continue;
		}
		let (node_a, node_b) =
			setup_two_nodes_with_store(&chain_source, false, true, false, store_config.store_type);
		let node_a = Arc::new(node_a);
		let node_b = Arc::new(node_b);

		runtime.block_on(async {
			let address_a = node_a.onchain_payment().new_address().unwrap();
			premine_and_distribute_funds(
				&bitcoind.client,
				&electrsd.client,
				vec![address_a],
				Amount::from_sat(35_000_000),
			)
			.await;
			node_a.sync_wallets().unwrap();
		});

		let node_a = Arc::clone(&node_a);
		let node_b = Arc::clone(&node_b);

		group.bench_function(store_config.name, |b| {
			b.iter_custom(|iter| {
				let node_a = Arc::clone(&node_a);
				let node_b = Arc::clone(&node_b);

				runtime.block_on(async {
					let mut total = Duration::ZERO;
					for _ in 0..iter {
						total += open_channel(
							Arc::clone(&node_a),
							Arc::clone(&node_b),
							&bitcoind.client,
							&electrsd,
						)
						.await;
					}
					total
				})
			});
		});
	}
}

fn startup_benchmark(c: &mut Criterion) {
	let (bitcoind, electrsd) = setup_bitcoind_and_electrsd();
	let chain_source = random_chain_source(&bitcoind, &electrsd);
	let runtime =
		tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();

	let mut group = c.benchmark_group("startup");
	group.sample_size(10);

	for store_config in store_bench_configs() {
		if !should_register_bench("startup", store_config.name) {
			continue;
		}
		let config = setup_startup_seed_node(
			&chain_source,
			&bitcoind,
			&electrsd,
			store_config.store_type,
			&runtime,
		);

		group.bench_function(store_config.name, |b| {
			b.iter_custom(|iter| {
				let mut total = Duration::ZERO;
				for _ in 0..iter {
					let start = Instant::now();
					let node = setup_node(&chain_source, config.clone());
					total += start.elapsed();
					node.stop().unwrap();
				}
				total
			});
		});
	}
}

fn setup_startup_seed_node(
	chain_source: &TestChainSource, bitcoind: &BitcoinD, electrsd: &electrsd::ElectrsD,
	store_type: TestStoreType, runtime: &tokio::runtime::Runtime,
) -> TestConfig {
	let mut config_a = random_config(true);
	config_a.store_type = store_type;
	let node_a = Arc::new(setup_node(chain_source, config_a.clone()));

	let mut config_b = random_config(true);
	config_b.store_type = store_type;
	let node_b = Arc::new(setup_node(chain_source, config_b));

	runtime.block_on(async {
		let address_a = node_a.onchain_payment().new_address().unwrap();
		premine_and_distribute_funds(
			&bitcoind.client,
			&electrsd.client,
			vec![address_a],
			Amount::from_sat(5_000_000),
		)
		.await;
		node_a.sync_wallets().unwrap();
		node_b.sync_wallets().unwrap();

		open_channel_push_amt(&node_a, &node_b, 1_000_000, Some(500_000_000), false, electrsd)
			.await;
		generate_blocks_and_wait(&bitcoind.client, &electrsd.client, 6).await;
		node_a.sync_wallets().unwrap();
		node_b.sync_wallets().unwrap();

		expect_channel_ready_event!(node_a, node_b.node_id());
		expect_channel_ready_event!(node_b, node_a.node_id());

		for description in ["startup seed 1", "startup seed 2"] {
			let invoice_description = Bolt11InvoiceDescription::Direct(
				Description::new(description.to_string()).unwrap(),
			);
			let invoice = node_b
				.bolt11_payment()
				.receive(1_000_000, &invoice_description.into(), 9217)
				.unwrap();
			let payment_id = node_a.bolt11_payment().send(&invoice, None).unwrap();
			wait_for_payment_success(&node_a, payment_id).await;
		}

		drain_events(&node_a);
		drain_events(&node_b);
	});

	node_a.stop().unwrap();
	node_b.stop().unwrap();

	config_a
}

fn should_register_bench(group: &str, name: &str) -> bool {
	let target = format!("{}/{}", group, name);
	let filters: Vec<String> =
		std::env::args().skip(1).filter(|arg| !arg.starts_with('-')).collect();
	filters.is_empty()
		|| filters.iter().any(|filter| {
			target.contains(filter) || (filter == group && target.starts_with(&format!("{group}/")))
		})
}

fn setup_forwarding_nodes(
	chain_source: &TestChainSource, bitcoind: &BitcoinD, electrsd: &electrsd::ElectrsD,
	store_type: TestStoreType, runtime: &tokio::runtime::Runtime,
) -> Vec<Arc<Node>> {
	let mut nodes = Vec::new();
	for _ in 0..5 {
		let mut config = random_config(true);
		config.store_type = store_type;
		nodes.push(Arc::new(setup_node(chain_source, config)));
	}

	runtime.block_on(async {
		let addresses =
			nodes.iter().map(|node| node.onchain_payment().new_address().unwrap()).collect();
		premine_and_distribute_funds(
			&bitcoind.client,
			&electrsd.client,
			addresses,
			Amount::from_sat(5_000_000),
		)
		.await;
		for node in &nodes {
			node.sync_wallets().unwrap();
		}

		let funding_amount_sat = 1_000_000;
		let push_amount_msat = None;
		open_channel_push_amt(
			&nodes[0],
			&nodes[1],
			funding_amount_sat,
			push_amount_msat,
			true,
			electrsd,
		)
		.await;
		open_channel_push_amt(
			&nodes[1],
			&nodes[2],
			funding_amount_sat,
			push_amount_msat,
			true,
			electrsd,
		)
		.await;
		nodes[1].sync_wallets().unwrap();
		open_channel_push_amt(
			&nodes[1],
			&nodes[3],
			funding_amount_sat,
			push_amount_msat,
			true,
			electrsd,
		)
		.await;
		open_channel_push_amt(
			&nodes[2],
			&nodes[4],
			funding_amount_sat,
			push_amount_msat,
			true,
			electrsd,
		)
		.await;
		open_channel_push_amt(
			&nodes[3],
			&nodes[4],
			funding_amount_sat,
			push_amount_msat,
			true,
			electrsd,
		)
		.await;

		generate_blocks_and_wait(&bitcoind.client, &electrsd.client, 6).await;
		for node in &nodes {
			node.sync_wallets().unwrap();
		}

		expect_event!(nodes[0], ChannelReady);
		expect_event!(nodes[1], ChannelReady);
		expect_event!(nodes[1], ChannelReady);
		expect_event!(nodes[1], ChannelReady);
		expect_event!(nodes[2], ChannelReady);
		expect_event!(nodes[2], ChannelReady);
		expect_event!(nodes[3], ChannelReady);
		expect_event!(nodes[3], ChannelReady);
		expect_event!(nodes[4], ChannelReady);
		expect_event!(nodes[4], ChannelReady);

		tokio::time::sleep(Duration::from_secs(1)).await;
		warm_up_forwarding_route(&nodes).await;
	});

	nodes
}

async fn send_forwarded_payments(nodes: Arc<Vec<Arc<Node>>>) -> Duration {
	let start = Instant::now();

	let total_payments = 1;
	let amount_msat = 2_500_000;
	let route_params = route_parameters();

	for _ in 0..total_payments {
		let invoice_description =
			Bolt11InvoiceDescription::Direct(Description::new("forwarding".to_string()).unwrap());
		let invoice = nodes[4]
			.bolt11_payment()
			.receive(amount_msat, &invoice_description.into(), 9217)
			.unwrap();
		let payment_id =
			nodes[0].bolt11_payment().send(&invoice, Some(route_params.clone())).unwrap();
		wait_for_forwarded_payment(&nodes, payment_id).await;
	}

	let duration = start.elapsed();

	for _ in 0..total_payments {
		let invoice_description =
			Bolt11InvoiceDescription::Direct(Description::new("return".to_string()).unwrap());
		let invoice = nodes[0]
			.bolt11_payment()
			.receive(amount_msat - 100_000, &invoice_description.into(), 9217)
			.unwrap();
		match nodes[4].bolt11_payment().send(&invoice, Some(route_params.clone())) {
			Ok(return_payment_id) => wait_for_payment_success(&nodes[4], return_payment_id).await,
			Err(_) => break,
		}
	}
	tokio::time::sleep(Duration::from_millis(10)).await;
	for node in nodes.iter() {
		drain_events(node);
	}

	duration
}

async fn wait_for_forwarded_payment(nodes: &[Arc<Node>], expected_payment_id: PaymentId) {
	let mut payment_successful = false;
	let mut first_hop_forwarded = false;
	let mut second_hop_forwarded = false;

	while !payment_successful || !first_hop_forwarded || !second_hop_forwarded {
		tokio::select! {
			event = nodes[0].next_event_async(), if !payment_successful => {
				match event {
					Event::PaymentSuccessful { payment_id: Some(payment_id), .. }
						if payment_id == expected_payment_id =>
					{
						payment_successful = true;
					},
					Event::PaymentFailed { payment_id, payment_hash, .. } => {
						nodes[0].event_handled().unwrap();
						panic!("Forwarded payment {:?} failed with hash {:?}", payment_id, payment_hash);
					},
					_ => {},
				}
				nodes[0].event_handled().unwrap();
			},
			event = nodes[1].next_event_async(), if !first_hop_forwarded => {
				if matches!(event, Event::PaymentForwarded { .. }) {
					first_hop_forwarded = true;
				}
				nodes[1].event_handled().unwrap();
			},
			event = nodes[2].next_event_async(), if !second_hop_forwarded => {
				if matches!(event, Event::PaymentForwarded { .. }) {
					second_hop_forwarded = true;
				}
				nodes[2].event_handled().unwrap();
			},
			event = nodes[3].next_event_async(), if !second_hop_forwarded => {
				if matches!(event, Event::PaymentForwarded { .. }) {
					second_hop_forwarded = true;
				}
				nodes[3].event_handled().unwrap();
			},
		}
	}
}

async fn warm_up_forwarding_route(nodes: &[Arc<Node>]) {
	for _ in 0..30 {
		let invoice_description = Bolt11InvoiceDescription::Direct(
			Description::new("forwarding warmup".to_string()).unwrap(),
		);
		let invoice = nodes[4]
			.bolt11_payment()
			.receive(2_500_000, &invoice_description.into(), 9217)
			.unwrap();
		if let Ok(payment_id) = nodes[0].bolt11_payment().send(&invoice, Some(route_parameters())) {
			wait_for_payment_success(&nodes[0], payment_id).await;
			tokio::time::sleep(Duration::from_millis(50)).await;
			for node in nodes {
				drain_events(node);
			}
			return;
		}
		tokio::time::sleep(Duration::from_secs(1)).await;
	}

	panic!("Timed out warming up forwarding route");
}

fn route_parameters() -> RouteParametersConfig {
	RouteParametersConfig {
		max_total_routing_fee_msat: Some(75_000),
		max_total_cltv_expiry_delta: 1000,
		max_path_count: 10,
		max_channel_saturation_power_of_half: 2,
	}
}

async fn open_channel(
	node_a: Arc<Node>, node_b: Arc<Node>, bitcoind: &BitcoindClient, electrsd: &electrsd::ElectrsD,
) -> Duration {
	let start = Instant::now();

	node_a
		.open_channel(
			node_b.node_id(),
			node_b.listening_addresses().unwrap().first().unwrap().clone(),
			100_000,
			None,
			None,
		)
		.unwrap();
	let duration = start.elapsed();

	assert!(node_a.list_peers().iter().any(|peer| peer.node_id == node_b.node_id()));

	let funding_txo_a = expect_channel_pending_event!(node_a, node_b.node_id());
	let funding_txo_b = expect_channel_pending_event!(node_b, node_a.node_id());
	assert_eq!(funding_txo_a, funding_txo_b);
	common::wait_for_tx(&electrsd.client, funding_txo_a.txid).await;

	generate_blocks_and_wait(bitcoind, &electrsd.client, 6).await;
	node_a.sync_wallets().unwrap();
	node_b.sync_wallets().unwrap();

	expect_channel_ready_event!(node_b, node_a.node_id());
	expect_channel_ready_event!(node_a, node_b.node_id());

	duration
}

async fn wait_for_payment_success(node: &Node, expected_payment_id: PaymentId) {
	loop {
		match node.next_event_async().await {
			Event::PaymentSuccessful { payment_id: Some(payment_id), .. }
				if payment_id == expected_payment_id =>
			{
				node.event_handled().unwrap();
				break;
			},
			Event::PaymentFailed { payment_id, payment_hash, .. } => {
				node.event_handled().unwrap();
				panic!("Return payment {:?} failed with hash {:?}", payment_id, payment_hash);
			},
			_ => node.event_handled().unwrap(),
		}
	}
}

fn drain_events(node: &Node) {
	while node.next_event().is_some() {
		node.event_handled().unwrap();
	}
}

fn store_bench_configs() -> Vec<StoreBenchConfig> {
	#[cfg(not(feature = "postgres"))]
	{
		vec![
			StoreBenchConfig { name: "sqlite", store_type: TestStoreType::Sqlite },
			StoreBenchConfig { name: "filesystem", store_type: TestStoreType::FilesystemStore },
		]
	}

	#[cfg(feature = "postgres")]
	{
		vec![
			StoreBenchConfig { name: "sqlite", store_type: TestStoreType::Sqlite },
			StoreBenchConfig { name: "filesystem", store_type: TestStoreType::FilesystemStore },
			StoreBenchConfig { name: "postgres", store_type: TestStoreType::Postgres },
		]
	}
}

criterion_group!(benches, operations_benchmark);
criterion_main!(benches);
