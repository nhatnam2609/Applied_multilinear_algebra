X = rand(10,10,10);
R = [2, 2, 2];
max_iter = 1000;
tol = 0.0001;
[ G_final, A ] = HOOI( X, R ,max_iter, tol);