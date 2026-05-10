#pragma once
#include "types.h"

Vector forwardSubstitution(const Matrix& L, const Vector& b);
Vector backwardSubstitutionLU(const Matrix& U, const Vector& y);

bool luDecomposition(const Matrix& A, Matrix& L, Matrix& U);
Vector solveWithLU(const Matrix& L, const Matrix& U, const Vector& b);
