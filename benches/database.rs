// This file is Copyright its original authors, visible in version control history.
//
// This file is licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. You may not use this file except in
// accordance with one or both of these licenses.

use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use ldk_node::bench::{
	configured_backends, payment_details_batch, payment_key, payment_update_batch_from_offset,
	pending_payment_details_batch_from_offset, pending_payment_update_batch_from_offset, Backend,
	PaginatedStoreFixture, StoreFixture, BATCH_LEN, PAGINATED_PAGE_LEN,
};

fn database_benchmark(c: &mut Criterion) {
	let backends = configured_backends();

	benchmark_payment_store_single_ops(c, &backends);
	benchmark_payment_store_warm_sequential(c, &backends);
	benchmark_payment_store_concurrent(c, &backends);
	benchmark_payment_store(c, &backends);
	benchmark_payment_store_paginated(c, &backends);
	benchmark_payment_store_lifecycle(c, &backends);
	benchmark_pending_payment_store(c, &backends);
}

fn benchmark_payment_store(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store");
	group.throughput(Throughput::Elements(BATCH_LEN));
	for backend in backends.iter().copied() {
		group.bench_with_input(
			BenchmarkId::new("insert_100_sequential_cold", backend.name()),
			&backend,
			|b, backend| {
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "payment_insert");
						let payments = payment_details_batch(0);
						(fixture, payments)
					},
					|(fixture, payments)| fixture.write_payment_batch(payments),
					BatchSize::SmallInput,
				)
			},
		);

		group.bench_with_input(
			BenchmarkId::new("update_100_sequential_cold", backend.name()),
			&backend,
			|b, backend| {
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "payment_update");
						fixture.write_payment_batch_from_offset(0);
						let updates = payment_update_batch_from_offset(0);
						(fixture, updates)
					},
					|(fixture, updates)| fixture.write_payment_update_batch(updates),
					BatchSize::SmallInput,
				)
			},
		);

		group.bench_with_input(
			BenchmarkId::new("reload_100_cold", backend.name()),
			&backend,
			|b, backend| {
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "payment_reload");
						fixture.write_payment_batch_from_offset(0);
						fixture
					},
					|fixture| {
						let payments = fixture.reload_payments();
						std::hint::black_box(payments);
					},
					BatchSize::SmallInput,
				)
			},
		);
	}
	group.finish();
}

fn benchmark_payment_store_paginated(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store_paginated");
	group.throughput(Throughput::Elements(PAGINATED_PAGE_LEN));
	for backend in backends.iter().copied() {
		let Some(fixture) = PaginatedStoreFixture::new(backend, "payment_list_page") else {
			continue;
		};
		group.bench_function(BenchmarkId::new("list_page_from_10k", backend.name()), |b| {
			b.iter(|| {
				let page_len = fixture.list_first_page();
				debug_assert_eq!(page_len, PAGINATED_PAGE_LEN as usize);
				std::hint::black_box(page_len);
			})
		});
		group.bench_function(BenchmarkId::new("list_second_page_from_10k", backend.name()), |b| {
			b.iter(|| {
				let page_len = fixture.list_second_page();
				debug_assert_eq!(page_len, PAGINATED_PAGE_LEN as usize);
				std::hint::black_box(page_len);
			})
		});
	}
	group.finish();
}

fn benchmark_pending_payment_store(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/pending_payment_store");
	group.throughput(Throughput::Elements(BATCH_LEN));
	for backend in backends.iter().copied() {
		group.bench_with_input(
			BenchmarkId::new("insert_100_sequential_cold", backend.name()),
			&backend,
			|b, backend| {
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "pending_payment_insert");
						let payments = pending_payment_details_batch_from_offset(0);
						(fixture, payments)
					},
					|(fixture, payments)| fixture.write_pending_payment_batch(payments),
					BatchSize::SmallInput,
				)
			},
		);

		group.bench_with_input(
			BenchmarkId::new("update_100_sequential_cold", backend.name()),
			&backend,
			|b, backend| {
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "pending_payment_update");
						let payments = pending_payment_details_batch_from_offset(0);
						fixture.write_pending_payment_batch(payments);
						let updates = pending_payment_update_batch_from_offset(0);
						(fixture, updates)
					},
					|(fixture, updates)| fixture.write_pending_payment_update_batch(updates),
					BatchSize::SmallInput,
				)
			},
		);
	}
	group.finish();
}

fn benchmark_payment_store_single_ops(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store_single");
	group.throughput(Throughput::Elements(1));
	for backend in backends.iter().copied() {
		let fixture = StoreFixture::new(backend, "single_write_new_key");
		let next_key = AtomicU64::new(0);
		group.bench_function(BenchmarkId::new("write_new_key", backend.name()), |b| {
			b.iter(|| {
				let idx = next_key.fetch_add(1, Ordering::Relaxed);
				fixture.write_payment(idx);
			})
		});

		let fixture = StoreFixture::new(backend, "single_update_existing_key");
		fixture.write_payment(0);
		group.bench_function(BenchmarkId::new("write_existing_key", backend.name()), |b| {
			b.iter(|| fixture.write_payment_update(0))
		});

		let fixture = StoreFixture::new(backend, "single_read_existing_key");
		fixture.write_payment(0);
		group.bench_function(BenchmarkId::new("read_existing_key", backend.name()), |b| {
			b.iter(|| {
				let payment = fixture.read_payment(0);
				std::hint::black_box(payment);
			})
		});

		group.bench_with_input(
			BenchmarkId::new("remove_existing_key", backend.name()),
			&backend,
			|b, backend| {
				let next_key = AtomicU64::new(0);
				b.iter_batched(
					|| {
						let fixture = StoreFixture::new(*backend, "single_remove_existing_key");
						let idx = next_key.fetch_add(1, Ordering::Relaxed);
						fixture.write_payment(idx);
						(fixture, payment_key(idx))
					},
					|(fixture, key)| fixture.remove_payment_key(&key),
					BatchSize::SmallInput,
				)
			},
		);
	}
	group.finish();
}

fn benchmark_payment_store_warm_sequential(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store_warm");
	group.throughput(Throughput::Elements(BATCH_LEN));
	for backend in backends.iter().copied() {
		let fixture = StoreFixture::new(backend, "payment_insert_100_sequential_warm");
		let next_offset = AtomicU64::new(0);
		group.bench_function(BenchmarkId::new("insert_100_sequential", backend.name()), |b| {
			b.iter(|| {
				let offset = next_offset.fetch_add(BATCH_LEN, Ordering::Relaxed);
				fixture.write_payment_batch_from_offset(offset);
			})
		});
	}
	group.finish();
}

fn benchmark_payment_store_concurrent(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store_concurrent");
	group.throughput(Throughput::Elements(BATCH_LEN));
	for backend in backends.iter().copied() {
		let fixture = StoreFixture::new(backend, "payment_insert_100_concurrent_distinct");
		let next_offset = AtomicU64::new(0);
		group.bench_function(BenchmarkId::new("insert_100_distinct_keys", backend.name()), |b| {
			b.iter(|| {
				let offset = next_offset.fetch_add(BATCH_LEN, Ordering::Relaxed);
				fixture.write_payment_batch_concurrent(offset, false);
			})
		});

		let fixture = StoreFixture::new(backend, "payment_insert_100_concurrent_same_key");
		group.bench_function(BenchmarkId::new("insert_100_same_key", backend.name()), |b| {
			b.iter(|| fixture.write_payment_batch_concurrent(0, true))
		});
	}
	group.finish();
}

fn benchmark_payment_store_lifecycle(c: &mut Criterion, backends: &[Backend]) {
	let mut group = c.benchmark_group("database/payment_store_lifecycle");
	group.throughput(Throughput::Elements(1));
	for backend in backends.iter().copied() {
		let fixture = StoreFixture::new(backend, "payment_lifecycle");
		let next_key = AtomicU64::new(0);
		group.bench_function(BenchmarkId::new("insert_update_read", backend.name()), |b| {
			b.iter(|| {
				let idx = next_key.fetch_add(1, Ordering::Relaxed);
				let payment = fixture.insert_update_read_payment(idx);
				std::hint::black_box(payment);
			})
		});
	}
	group.finish();
}

criterion_group!(benches, database_benchmark);
criterion_main!(benches);
