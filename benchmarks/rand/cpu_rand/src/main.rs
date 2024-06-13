use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

fn parallel(v: &mut [f32]) {
    v.par_iter_mut()
        .for_each_init(SmallRng::from_entropy, |rng, x| {
            *x = rng.gen();
        });
}

fn sequential(v: &mut [f32]) {
    let mut rng = SmallRng::from_entropy();
    v.iter_mut().for_each(|x| {
        *x = rng.gen();
    });
}

fn main() {
    const N: usize = 10_000_000;
    const ITERS: usize = 1000;

    let mut v: Vec<f32> = vec![0.0; N];
    sequential(&mut v);
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        sequential(&mut v);
    }
    println!("{}", v[2137]);
    let elapsed = t0.elapsed();
    println!(
        "GSamples/sec {:.1}",
        ITERS as f64 * N as f64 / elapsed.as_secs_f64() / 1e9
    )
}
