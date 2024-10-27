X = rand([5,4,3]);
R = 3;
tol = 0.000001;
max_iter = 1000;

[lambda, A,B,C] = CP_ALS(X, R, max_iter, tol);


