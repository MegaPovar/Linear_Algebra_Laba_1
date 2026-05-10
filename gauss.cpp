#include "gauss.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

const double EPS = 1e-18;

Vector backwardSubstitution(const Matrix& U, const Vector& y) {
    int n = (int)U.size();
    Vector x(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= U[i][j] * x[j];
        }

        if (std::abs(U[i][i]) < EPS) {
            throw std::runtime_error("Zero diagonal element in backward substitution");
        }

        x[i] = sum / U[i][i];
    }

    return x;
}

Vector gaussNoPivot(Matrix A, Vector b) {
    int n = (int)A.size();

    for (int k = 0; k < n; ++k) {
        if (std::abs(A[k][k]) < EPS) {
            throw std::runtime_error("Zero pivot in Gauss without pivoting");
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    return backwardSubstitution(A, b);
}

Vector gaussPartialPivot(Matrix A, Vector b) {
    int n = (int)A.size();

    for (int k = 0; k < n; ++k) {
        int pivotRow = k;
        double maxVal = std::abs(A[k][k]);

        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > maxVal) {
                maxVal = std::abs(A[i][k]);
                pivotRow = i;
            }
        }

        if (std::abs(A[pivotRow][k]) < EPS) {
            throw std::runtime_error("Matrix is singular in Gauss with partial pivoting");
        }

        if (pivotRow != k) {
            std::swap(A[k], A[pivotRow]);
            std::swap(b[k], b[pivotRow]);
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    return backwardSubstitution(A, b);
}
