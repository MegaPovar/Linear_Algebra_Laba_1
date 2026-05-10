#include "lu.h"
#include <cmath>
#include <stdexcept>

const double EPS = 1e-18;

Vector forwardSubstitution(const Matrix& L, const Vector& b) {
    int n = (int)L.size();
    Vector y(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= L[i][j] * y[j];
        }

        if (std::abs(L[i][i]) < EPS) {
            throw std::runtime_error("Zero diagonal element in forward substitution");
        }

        y[i] = sum / L[i][i];
    }

    return y;
}

Vector backwardSubstitutionLU(const Matrix& U, const Vector& y) {
    int n = (int)U.size();
    Vector x(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= U[i][j] * x[j];
        }

        if (std::abs(U[i][i]) < EPS) {
            throw std::runtime_error("Zero diagonal element in LU backward substitution");
        }

        x[i] = sum / U[i][i];
    }

    return x;
}

bool luDecomposition(const Matrix& A, Matrix& L, Matrix& U) {
    int n = (int)A.size();

    L.assign(n, Vector(n, 0.0));
    U.assign(n, Vector(n, 0.0));

    for (int i = 0; i < n; ++i) {
        L[i][i] = 1.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = A[i][j] - sum;
        }

        if (std::abs(U[i][i]) < EPS) {
            return false;
        }

        for (int j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[j][k] * U[k][i];
            }
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }

    return true;
}

Vector solveWithLU(const Matrix& L, const Matrix& U, const Vector& b) {
    Vector y = forwardSubstitution(L, b);
    return backwardSubstitutionLU(U, y);
}
