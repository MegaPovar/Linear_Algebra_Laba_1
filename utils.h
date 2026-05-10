#pragma once
#include "types.h"

#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "gauss.h"
#include "lu.h"

// Базовые функции

inline Matrix identityMatrix(int n) {
    Matrix I(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

inline double vectorNorm(const Vector& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

inline Vector matVecMul(const Matrix& A, const Vector& x) {
    int n = (int)A.size();
    Vector res(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (int)A[i].size(); ++j) {
            res[i] += A[i][j] * x[j];
        }
    }

    return res;
}

inline double residualNorm(const Matrix& A, const Vector& x, const Vector& b) {
    Vector Ax = matVecMul(A, x);
    Vector r(b.size());

    for (int i = 0; i < (int)b.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }

    return vectorNorm(r); 
}

inline double relativeError(const Vector& xApprox, const Vector& xExact) {
    Vector diff(xExact.size());

    for (int i = 0; i < (int)xExact.size(); ++i) {
        diff[i] = xApprox[i] - xExact[i];
    }

    double denom = vectorNorm(xExact);
    if (denom < 1e-12) {
        return vectorNorm(diff);
    }

    return vectorNorm(diff) / denom;
}

// Генерация данных

inline Matrix generateRandomMatrix(int n, std::mt19937& rng, double left = -1.0, double right = 1.0) {
    std::uniform_real_distribution<double> dist(left, right);
    Matrix A(n, Vector(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = dist(rng);
        }
    }

    return A;
}

inline Vector generateRandomVector(int n, std::mt19937& rng, double left = -1.0, double right = 1.0) {
    std::uniform_real_distribution<double> dist(left, right);
    Vector b(n);

    for (int i = 0; i < n; ++i) {
        b[i] = dist(rng);
    }

    return b;
}

inline Matrix generateHilbertMatrix(int n) {
    Matrix H(n, Vector(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i][j] = 1.0 / (i + j + 1.0);
        }
    }

    return H;
}

inline Matrix makeDiagonallyDominant(Matrix A) {
    int n = (int)A.size();

    for (int i = 0; i < n; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                rowSum += std::abs(A[i][j]);
            }
        }
        A[i][i] = rowSum + 1.0;
    }

    return A;
}

// Замер времени

template<typename Func, typename... Args>
double measureTimeMs(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = finish - start;
    return elapsed.count();
}

// Эксперименты

inline void experimentSingleSystem(std::mt19937& rng) {
    std::cout << "\n================ EXPERIMENT 1: SINGLE SYSTEM ================\n";
    std::cout << "Time is measured in milliseconds.\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(18) << "Gauss no pivot"
              << std::setw(18) << "Gauss pivot"
              << std::setw(18) << "LU decomp"
              << std::setw(18) << "LU solve"
              << std::setw(18) << "LU total"
              << "\n";

    std::vector<int> sizes = {100, 200, 500, 1000};

    for (int n : sizes) {
        Matrix A = makeDiagonallyDominant(generateRandomMatrix(n, rng));
        Vector b = generateRandomVector(n, rng);

        double tGaussNoPivot = 0.0;
        double tGaussPivot = 0.0;
        double tLUDecomp = 0.0;
        double tLUSolve = 0.0;
        double tLUTotal = 0.0;

        try {
            tGaussNoPivot = measureTimeMs(gaussNoPivot, A, b);
        } catch (...) {
            tGaussNoPivot = -1.0;
        }

        try {
            tGaussPivot = measureTimeMs(gaussPartialPivot, A, b);
        } catch (...) {
            tGaussPivot = -1.0;
        }

        try {
            Matrix L, U;

            auto start1 = std::chrono::high_resolution_clock::now();
            bool ok = luDecomposition(A, L, U);
            auto end1 = std::chrono::high_resolution_clock::now();

            if (!ok) {
                throw std::runtime_error("LU decomposition failed");
            }

            auto start2 = std::chrono::high_resolution_clock::now();
            Vector x = solveWithLU(L, U, b);
            (void)x;
            auto end2 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> d1 = end1 - start1;
            std::chrono::duration<double, std::milli> d2 = end2 - start2;

            tLUDecomp = d1.count();
            tLUSolve = d2.count();
            tLUTotal = tLUDecomp + tLUSolve;
        } catch (...) {
            tLUDecomp = -1.0;
            tLUSolve = -1.0;
            tLUTotal = -1.0;
        }

        std::cout << std::left
                  << std::setw(8)  << n
                  << std::setw(18) << tGaussNoPivot
                  << std::setw(18) << tGaussPivot
                  << std::setw(18) << tLUDecomp
                  << std::setw(18) << tLUSolve
                  << std::setw(18) << tLUTotal
                  << "\n";
    }
}

inline void experimentMultipleRightParts(std::mt19937& rng) {
    std::cout << "\n================ EXPERIMENT 2: MULTIPLE RIGHT-HAND SIDES ================\n";
    std::cout << "Fixed matrix size: n = 500. Time is measured in milliseconds.\n";

    int n = 500;
    Matrix A = makeDiagonallyDominant(generateRandomMatrix(n, rng));
    std::vector<int> ks = {1, 10, 100};

    std::cout << std::left
              << std::setw(8)  << "k"
              << std::setw(22) << "Gauss pivot total"
              << std::setw(22) << "LU total"
              << std::setw(22) << "LU decomp only"
              << std::setw(22) << "LU solves only"
              << "\n";

    for (int k : ks) {
        std::vector<Vector> bs;
        for (int i = 0; i < k; ++i) {
            bs.push_back(generateRandomVector(n, rng));
        }

        double gaussTotal = 0.0;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < k; ++i) {
                gaussPartialPivot(A, bs[i]);
            }
            auto finish = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> elapsed = finish - start;
            gaussTotal = elapsed.count();
        } catch (...) {
            gaussTotal = -1.0;
        }

        double luTotal = 0.0;
        double luDecompOnly = 0.0;
        double luSolvesOnly = 0.0;

        try {
            Matrix L, U;

            auto start1 = std::chrono::high_resolution_clock::now();
            bool ok = luDecomposition(A, L, U);
            auto end1 = std::chrono::high_resolution_clock::now();

            if (!ok) {
                throw std::runtime_error("LU decomposition failed");
            }

            auto start2 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < k; ++i) {
                Vector x = solveWithLU(L, U, bs[i]);
                (void)x;
            }
            auto end2 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> d1 = end1 - start1;
            std::chrono::duration<double, std::milli> d2 = end2 - start2;

            luDecompOnly = d1.count();
            luSolvesOnly = d2.count();
            luTotal = luDecompOnly + luSolvesOnly;
        } catch (...) {
            luTotal = -1.0;
            luDecompOnly = -1.0;
            luSolvesOnly = -1.0;
        }

        std::cout << std::left
                  << std::setw(8)  << k
                  << std::setw(22) << gaussTotal
                  << std::setw(22) << luTotal
                  << std::setw(22) << luDecompOnly
                  << std::setw(22) << luSolvesOnly
                  << "\n";
    }
}

inline void experimentHilbert() {
    std::cout << "\n================ EXPERIMENT 3: HILBERT MATRIX ================\n";
    std::cout << "The exact solution is x = (1, 1, ..., 1)^T.\n";
    std::cout << std::scientific << std::setprecision(6);

    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(22) << "Method"
              << std::setw(22) << "Rel. error"
              << std::setw(22) << "Residual"
              << "\n";

    std::vector<int> sizes = {5, 10, 15};

    for (int n : sizes) {
        Matrix H = generateHilbertMatrix(n);
        Vector xExact(n, 1.0);
        Vector b = matVecMul(H, xExact);

        try {
            Vector x = gaussNoPivot(H, b);
            double err = relativeError(x, xExact);
            double res = residualNorm(H, x, b);

            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "Gauss no pivot"
                      << std::setw(22) << err
                      << std::setw(22) << res
                      << "\n";
        } catch (...) {
            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "Gauss no pivot"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }

        try {
            Vector x = gaussPartialPivot(H, b);
            double err = relativeError(x, xExact);
            double res = residualNorm(H, x, b);

            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "Gauss pivot"
                      << std::setw(22) << err
                      << std::setw(22) << res
                      << "\n";
        } catch (...) {
            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "Gauss pivot"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }

        try {
            Matrix L, U;
            bool ok = luDecomposition(H, L, U);
            if (!ok) {
                throw std::runtime_error("LU decomposition failed");
            }

            Vector x = solveWithLU(L, U, b);
            double err = relativeError(x, xExact);
            double res = residualNorm(H, x, b);

            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "LU"
                      << std::setw(22) << err
                      << std::setw(22) << res
                      << "\n";
        } catch (...) {
            std::cout << std::left
                      << std::setw(8)  << n
                      << std::setw(22) << "LU"
                      << std::setw(22) << "FAILED"
                      << std::setw(22) << "FAILED"
                      << "\n";
        }
    }
}

inline void runAllExperiments() {
    std::cout << std::fixed << std::setprecision(8);

    const unsigned int SEED = 42;
    std::mt19937 rng(SEED);

    experimentSingleSystem(rng);
    experimentMultipleRightParts(rng);
    experimentHilbert();
}
