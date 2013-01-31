
#ifndef __EXACT_SOLUTION__
#define __EXACT_SOLUTION__

/*
   inline long double solution1D(long double x) {
   long double res = sin(__PI__ * x);
   return res;
   }

   inline long double solutionDerivative1D(long double x, int dofX) {
   long double res = myIntPow(__PI__, dofX);
   if(((dofX/2)%2) != 0) {
   res *= -1;
   }
   if((dofX%2) == 0) {
   res *= sin(__PI__ * x);
   } else {
   res *= cos(__PI__ * x);
   }
   return res;
   }
   */

inline long double solution1D(long double x) {
  long double res = exp((x - 0.5)*(x - 0.5));
  return res;
}

inline long double solutionDerivative1D(long double x, int dofX) {
  long double res;
  if(dofX == 0) {
    res = solution1D(x);
  } else if(dofX == 1) {
    res = 2.0 * (x - 0.5) * solution1D(x);
  } else if(dofX == 2) {
    res = (2.0 * solution1D(x)) + (2.0 * (x - 0.5) * solutionDerivative1D(x, 1));
  } else if(dofX == 3) {
    res = (4.0 * solutionDerivative1D(x, 1)) + (2.0 * (x - 0.5) * solutionDerivative1D(x, 2));
  } else if(dofX == 4) {
    res = (6.0 * solutionDerivative1D(x, 2)) + (2.0 * (x - 0.5) * solutionDerivative1D(x, 3));
  } else {
    res = ((static_cast<long double>(2*(dofX - 1))) * solutionDerivative1D(x, (dofX - 2))) +
      (2.0 * (x - 0.5) * solutionDerivative1D(x, (dofX - 1)));
  }
  return res;
}

inline long double solution2D(long double x, long double y) {
  long double res = solution1D(x) * solution1D(y);
  return res;
}

inline long double solution3D(long double x, long double y, long double z) {
  long double res = solution1D(x) * solution1D(y) * solution1D(z);
  return res;
}

inline long double solutionDerivative2D(long double x, long double y, int dofX, int dofY) {
  long double res = solutionDerivative1D(x, dofX) * solutionDerivative1D(y, dofY);
  return res;
}

inline long double solutionDerivative3D(long double x, long double y, long double z, int dofX, int dofY, int dofZ) {
  long double res = solutionDerivative1D(x, dofX) * solutionDerivative1D(y, dofY) * solutionDerivative1D(z, dofZ);
  return res;
}

#endif



