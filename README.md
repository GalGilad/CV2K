# CV2K
An automated approach for determining the number of components in non-negative matrix factorization

How to run (examples):
- python CV2K_main.py --data file_name.npy --fraction 0.1
- For fraction-free variant: CV2K_main.py --version x --data file_name.npy

List of parameters:
- --version: standard / x; default=standard; standard variant vs fraction-free variant
- --auto: y / n; default=n; automatic K search scheme vs using [bottom_k, top_k] range
- --data: file_name.npy; n x m catalog: n rows are samples, m columns are mutation types
- --fraction: default=0.1; validation fraction; not used in fraction-free variant
- --reps: default=30; number of repetitions per rank. rollback scheme is less reliable when number of reps is small
- --maxiter: default=2000; max number of iterations
- --bottom_k: default=1; run from bottom_k
- --top_k: default=10; to top_k
- --stride: default=1; for example, if --bottom_k 1 --top_k 5 --stride 2, then range is 1-3-5
- --workers: default=20; number of workers
- --obj: kl / euc / is; default=kl, KL-divergence, euclidean, Itakura-Saito

Real data extracted from:
https://docs.icgc.org/

Simulated data from:
https://www.synapse.org/#!Synapse:syn11726601/files/

L. Alexandrov, J. Kim, N. Haradhvala, M. Huang, A. Ng, Y. Wu, A. Boot,
K. Covington, D. Gordenin, E. Bergstrom, S. Islam, N. L´opez-Bigas,
L. Klimczak, J. McPherson, S. Morganella, R. Sabarinathan, D. Wheeler,
V. Mustonen, P. Boutros, and W. Yu. The repertoire of mutational signatures in human cancer. Nature, 578:94–101, 02 2020.
