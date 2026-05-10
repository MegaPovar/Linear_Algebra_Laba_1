#pragma once
#include "types.h"

Vector backwardSubstitution(const Matrix& U, const Vector& y);
Vector gaussNoPivot(Matrix A, Vector b);
Vector gaussPartialPivot(Matrix A, Vector b);
