.PHONY: install test \
	repro-mnist_ae_compression repro-mnist_ae_sompression \
	repro-mnist_betta_vae repro-mnist_f_vae \
	repro-mnist_gan_comparison repro-mnist_gan_extensions \
	repro-mnist_pg_gan repro-mnist_realnvp repro-mnist_vae_comparison

install:
	uv sync

test:
	uv run pytest

repro-mnist_ae_compression:
	uv run dvc repro mnist_ae_compression_eval

repro-mnist_betta_vae:
	uv run dvc repro mnist_betta_vae_eval mnist_betta_vae_explore

repro-mnist_f_vae:
	uv run dvc repro mnist_f_vae_eval mnist_f_vae_eval_fashion

repro-mnist_gan_comparison:
	uv run dvc repro mnist_gan_comparison_eval

repro-mnist_gan_extensions:
	uv run dvc repro mnist_gan_extensions_eval

repro-mnist_pg_gan:
	uv run dvc repro mnist_pg_gan_eval

repro-mnist_realnvp:
	uv run dvc repro mnist_realnvp_eval mnist_realnvp_eval_fashion

repro-mnist_vae_comparison:
	uv run dvc repro mnist_vae_comparison_eval
