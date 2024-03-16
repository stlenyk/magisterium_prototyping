Mix.install(
  [
    {:nx, "~> 0.7.1"},
    {:exla, "~> 0.7.1"},
    {:number, "~> 1.0.4"}
  ],
  config: [
    nx: [default_backend: EXLA.Backend]
  ],
  system_env: [
    XLA_TARGET: "cuda120"
  ]
)

n = 1000
iters = 10
shape = {n, n}
key = Nx.Random.key(2137)
{t1, key} = Nx.Random.uniform(key, shape: shape)
{t2, _key} = Nx.Random.uniform(key, shape: shape)

calculation = fn ->
  Enum.map(0..iters, fn _ -> Nx.dot(t1, t2) |> Nx.sum(axes: [0, 1]) end)
end

{time, res} = :timer.tc(calculation, [])
IO.inspect(res)
~s/#{(time / 1000 / iters) |> Number.Delimit.number_to_delimited()} ms\/iter/ |> IO.puts()
