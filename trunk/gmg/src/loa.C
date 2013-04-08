
#include "gmg/include/loa.h"
#include "gmg/include/boundary.h"
#include "common/include/commonUtils.h"
#include <algorithm>
#include <cmath>

void setupLOA(LOAdata* data, int K, DM daL, DM daH,
    std::vector<std::vector<long long int> >& coeffs) {
  data->K = K;
  data->coeffs = &coeffs;
  data->daL = daL;
  data->daH = daH;
}

void destroyLOA(LOAdata* data) {
  delete data;
}

void applyLOA(LOAdata* data, Vec high, Vec low) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  PetscInt dofsPerNode;
  DMDAGetInfo((data->daH), &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners((data->daH), &xs, &ys, &zs, &nx, &ny, &nz);

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy = 0;
  if(dim < 2) {
    ys = 0;
    ny = 1;
    Ny = 1;
  } else {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
  if(dim < 3) {
    zs = 0;
    nz = 1;
    Nz = 1;
  } else {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  std::vector<int> pStar;
  computePstar((data->daH), high, pStar);

  std::vector<double> aStar;
  std::vector<double> cStar;
  if(dim == 1) {
    aStar.resize(pStar.size(), 0.0);
    cStar.resize(aStar.size(), hx);
  } else if(dim == 2) {
    aStar.resize((pStar.size()/2), 0.0);
    cStar.resize(aStar.size(), std::sqrt((hx*hx) + (hy*hy)));
  } else {
    aStar.resize((pStar.size()/3), 0.0);
    cStar.resize(aStar.size(), std::sqrt((hx*hx) + (hy*hy) + (hz*hz)));
  }

  Vec loc;
  DMGetLocalVector((data->daH), &loc);

  DMGlobalToLocalBegin((data->daH), high, INSERT_VALUES, loc);
  DMGlobalToLocalEnd((data->daH), high, INSERT_VALUES, loc);

  if(dim == 1) {
    PetscScalar** arr = NULL;
    DMDAVecGetArrayDOF((data->daH), loc, &arr);
    for(size_t i = 0; i < aStar.size(); ++i) {
      int xSt = pStar[i];
      int xs = xSt - 1;
      int xe = xSt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      std::vector<double> fVec; 
      for(int xi = xs; xi <= xe; ++xi) {
        for(int d = 0; d < dofsPerNode; ++d) {
          fVec.push_back(arr[xi][d]);
        }//end d
      }//end xi
      std::vector<double> fHat(fVec.size());
      std::vector<double> rVec(fVec.size());
      std::vector<double> gradFhat(fVec.size());
      double cSqr = cStar[i] * cStar[i];
      for(size_t j = 0; j < fHat.size(); ++j) {
        fHat[j] = 0;
      }//end j
      double aNum = 0.0;
      double aDen = 0.0;
      for(size_t j = 0; j < fVec.size(); ++j) {
        aNum += (fHat[j] * fVec[j]);
        aDen += (fHat[j] * fHat[j]);
      }//end j
      aStar[i] = aNum/aDen;
      for(size_t j = 0; j < fVec.size(); ++j) {
        rVec[j] = (aStar[i] * fHat[j]) - fVec[j];
      }//end j
      double oHat = 0;
      for(size_t j = 0; j < rVec.size(); ++j) {
        oHat += (rVec[j] * rVec[j]);
      }//end j
      oHat *= 0.5;
      for(int iter = 0; iter < 100; ++iter) {
        if(oHat <= 1.0e-12) {
          break;
        }
        for(size_t j = 0; j < gradFhat.size(); ++j) {
          gradFhat[j] = 0;
        }//end j
        double term1 = 0;
        double term2 = 0;
        for(size_t j = 0; j < fVec.size(); ++j) {
          term1 += (gradFhat[j] * fVec[j]);
          term2 += (fHat[j] * gradFhat[j]);
        }//end j
        double gradA = ((aDen * term1) - (2.0 * term2 * aNum))/(aDen * aDen);
        std::vector<double> jVec(fVec.size());
        for(size_t j = 0; j < fVec.size(); ++j) {
          jVec[j] = (gradA * fHat[j]) + (aStar[i] * gradFhat[j]);
        }//end j
        double gradO = 0;
        for(size_t j = 0; j < rVec.size(); ++j) {
          gradO += (rVec[j] * jVec[j]);
        }//end j
        if(fabs(gradO) <= 1.0e-12) {
          break;
        }
        double hessO = 0;
        for(size_t j = 0; j < jVec.size(); ++j) {
          hessO += (jVec[j] * jVec[j]);
        }//end j
        double step = -gradO/hessO;
        if(fabs(step) <= 1.0e-12) {
          break;
        }
        double alpha = 1.0;
        while(alpha > 1.0e-12) {
          double tmp = cStar[i] + (alpha * step); 
          cSqr = tmp * tmp;
          double tmpAstar = aNum/aDen;
          if(tmpObj < oHat) {
            oHat = tmpObj;
            cStar[i] = tmp;
            aStar[i] = tmpAstar;
            break;
          } else {
            alpha *= 0.5;
          }
        }//end while
        if(alpha <= 1.0e-12) {
          break;
        }
      }//end iter
    }//end i
    DMDAVecRestoreArrayDOF((data->daH), loc, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr = NULL;
    DMDAVecGetArrayDOF((data->daH), loc, &arr);
    for(size_t i = 0; i < aStar.size(); ++i) {
      int xSt = pStar[2*i];
      int ySt = pStar[(2*i) + 1];
      int xs = xSt - 1;
      int xe = xSt + 1;
      int ys = ySt - 1;
      int ye = ySt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      if(ys < 0) {
        ys = 0;
      }
      if(ye >= Ny) {
        ye = Ny - 1;
      }
      std::vector<double> fVec; 
      for(int yi = ys; yi <= ye; ++yi) {
        for(int xi = xs; xi <= xe; ++xi) {
          for(int d = 0; d < dofsPerNode; ++d) {
            fVec.push_back(arr[yi][xi][d]);
          }//end d
        }//end xi
      }//end yi
      std::vector<double> fHat(fVec.size());
      double cSqr = cStar[i] * cStar[i];
      for(size_t j = 0; j < fHat.size(); ++j) {
        fHat[j] = 0;
      }//end j
      double aNum = 0.0;
      double aDen = 0.0;
      for(size_t j = 0; j < fVec.size(); ++j) {
        aNum += (fHat[j] * fVec[j]);
        aDen += (fHat[j] * fHat[j]);
      }//end j
      aStar[i] = aNum/aDen;
    }//end i
    DMDAVecRestoreArrayDOF((data->daH), loc, &arr);
  } else {
    PetscScalar**** arr = NULL;
    DMDAVecGetArrayDOF((data->daH), loc, &arr);
    for(size_t i = 0; i < aStar.size(); ++i) {
      int xSt = pStar[3*i];
      int ySt = pStar[(3*i) + 1];
      int zSt = pStar[(3*i) + 2];
      int xs = xSt - 1;
      int xe = xSt + 1;
      int ys = ySt - 1;
      int ye = ySt + 1;
      int zs = zSt - 1;
      int ze = zSt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      if(ys < 0) {
        ys = 0;
      }
      if(ye >= Ny) {
        ye = Ny - 1;
      }
      if(zs < 0) {
        zs = 0;
      }
      if(ze >= Nz) {
        ze = Nz - 1;
      }
      std::vector<double> fVec; 
      for(int zi = zs; zi <= ze; ++zi) {
        for(int yi = ys; yi <= ye; ++yi) {
          for(int xi = xs; xi <= xe; ++xi) {
            for(int d = 0; d < dofsPerNode; ++d) {
              fVec.push_back(arr[zi][yi][xi][d]);
            }//end d
          }//end xi
        }//end yi
      }//end zi
      std::vector<double> fHat(fVec.size());
      double cSqr = cStar[i] * cStar[i];
      for(size_t j = 0; j < fHat.size(); ++j) {
        fHat[j] = 0;
      }//end j
      double aNum = 0.0;
      double aDen = 0.0;
      for(size_t j = 0; j < fVec.size(); ++j) {
        aNum += (fHat[j] * fVec[j]);
        aDen += (fHat[j] * fHat[j]);
      }//end j
      aStar[i] = aNum/aDen;
    }//end i
    DMDAVecRestoreArrayDOF((data->daH), loc, &arr);
  }

  DMRestoreLocalVector((data->daH), &loc);

  computeFhat((data->daL), ((data->K) - 1), (*(data->coeffs))[((data->K) - 1)], pStar,
      aStar, cStar, low);
}

void computeFhat(DM da, int K, std::vector<long long int>& coeffs, std::vector<int>& pStar,
    std::vector<double>& aStar, std::vector<double>& cStar, Vec out) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy = 0;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  PetscInt extraNumGpts = 0;
  //PetscOptionsGetInt(PETSC_NULL, "-extraGptsFhat", &extraNumGpts, PETSC_NULL);
  int numGaussPts = (2*K) + 2 + extraNumGpts;

  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  std::vector<std::vector<std::vector<long double> > > shFnVals(2);
  for(int node = 0; node < 2; ++node) {
    shFnVals[node].resize(K + 1);
    for(int dof = 0; dof <= K; ++dof) {
      (shFnVals[node][dof]).resize(numGaussPts);
      for(int g = 0; g < numGaussPts; ++g) {
        shFnVals[node][dof][g] = eval1DshFn(node, dof, K, coeffs, gPt[g]);
      }//end g
    }//end dof
  }//end node

  Vec loc;
  DMGetLocalVector(da, &loc);

  VecZeroEntries(loc);

  PetscScalar** arr1d = NULL;
  PetscScalar*** arr2d = NULL;
  PetscScalar**** arr3d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, loc, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, loc, &arr2d);
  } else {
    DMDAVecGetArrayDOF(da, loc, &arr3d);
  }

  if(dim == 1) {
    for(size_t i = 0; i < aStar.size(); ++i) {
      double cSqr = cStar[i] * cStar[i];
      int xSt = pStar[i];
      int xs = xSt - 1;
      int xe = xSt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      for(int xi = xs; xi < xe; ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(int node = 0; node < 2; ++node) {
          for(int dof = 0; dof <= K; ++dof) {
            for(int g = 0; g < numGaussPts; ++g) {
              long double xg = coordLocalToGlobal(gPt[g], xa, hx);
              int deltaX = xg - xSt;
              double denom = (deltaX*deltaX) - (hx*hx);
              double force = std::exp(-cSqr/denom);
              arr1d[xi + node][dof] += ( gWt[g] * shFnVals[node][dof][g] * force );
            }//end g
          }//end dof
        }//end node
      }//end xi
    }//end i
  } else if(dim == 2) {
    for(size_t i = 0; i < aStar.size(); ++i) {
      double cSqr = cStar[i] * cStar[i];
      int xSt = pStar[2*i];
      int ySt = pStar[(2*i) + 1];
      int xs = xSt - 1;
      int xe = xSt + 1;
      int ys = ySt - 1;
      int ye = ySt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      if(ys < 0) {
        ys = 0;
      }
      if(ye >= Ny) {
        ye = Ny - 1;
      }
      for(int yi = ys; yi < ye; ++yi) {
        long double ya = (static_cast<long double>(yi))*hy;
        for(int xi = xs; xi < xe; ++xi) {
          long double xa = (static_cast<long double>(xi))*hx;
          for(int nodeY = 0; nodeY < 2; ++nodeY) {
            for(int nodeX = 0; nodeX < 2; ++nodeX) {
              for(int dofY = 0, d = 0; dofY <= K; ++dofY) {
                for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                  for(int gY = 0; gY < numGaussPts; ++gY) {
                    long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                    int deltaY = yg - ySt;
                    for(int gX = 0; gX < numGaussPts; ++gX) {
                      long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                      int deltaX = xg - xSt;
                      double denom = (deltaX*deltaX) + (deltaY*deltaY) - (hx*hx) - (hy*hy);
                      double force = std::exp(-cSqr/denom);
                      arr2d[yi + nodeY][xi + nodeX][d] += ( gWt[gX] * gWt[gY] 
                          * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] * force );
                    }//end gX
                  }//end gY
                }//end dofX
              }//end dofY
            }//end nodeX
          }//end nodeY
        }//end xi
      }//end yi
    }//end i
  } else {
    for(size_t i = 0; i < aStar.size(); ++i) {
      double cSqr = cStar[i] * cStar[i];
      int xSt = pStar[3*i];
      int ySt = pStar[(3*i) + 1];
      int zSt = pStar[(3*i) + 2];
      int xs = xSt - 1;
      int xe = xSt + 1;
      int ys = ySt - 1;
      int ye = ySt + 1;
      int zs = zSt - 1;
      int ze = zSt + 1;
      if(xs < 0) {
        xs = 0;
      }
      if(xe >= Nx) {
        xe = Nx - 1;
      }
      if(ys < 0) {
        ys = 0;
      }
      if(ye >= Ny) {
        ye = Ny - 1;
      }
      if(zs < 0) {
        zs = 0;
      }
      if(ze >= Nz) {
        ze = Nz - 1;
      }
      for(int zi = zs; zi < ze; ++zi) {
        long double za = (static_cast<long double>(zi))*hz;
        for(int yi = ys; yi < ye; ++yi) {
          long double ya = (static_cast<long double>(yi))*hy;
          for(int xi = xs; xi < xe; ++xi) {
            long double xa = (static_cast<long double>(xi))*hx;
            for(int nodeZ = 0; nodeZ < 2; ++nodeZ) {
              for(int nodeY = 0; nodeY < 2; ++nodeY) {
                for(int nodeX = 0; nodeX < 2; ++nodeX) {
                  for(int dofZ = 0, d = 0; dofZ <= K; ++dofZ) {
                    for(int dofY = 0; dofY <= K; ++dofY) {
                      for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                        for(int gZ = 0; gZ < numGaussPts; ++gZ) {
                          long double zg = coordLocalToGlobal(gPt[gZ], za, hz);
                          int deltaZ = zg - zSt;
                          for(int gY = 0; gY < numGaussPts; ++gY) {
                            long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                            int deltaY = yg - ySt;
                            for(int gX = 0; gX < numGaussPts; ++gX) {
                              long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                              int deltaX = xg - xSt;
                              double denom = (deltaX*deltaX) + (deltaY*deltaY) + (deltaZ*deltaZ) 
                                - (hx*hx) - (hy*hy) - (hz*hz);
                              double force = std::exp(-cSqr/denom);
                              arr3d[zi + nodeZ][yi + nodeY][xi + nodeX][d] += ( gWt[gX] * gWt[gY] * gWt[gZ] 
                                  * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] * shFnVals[nodeZ][dofZ][gZ]
                                  * force );
                            }//end gX
                          }//end gY
                        }//end gZ
                      }//end dofX
                    }//end dofY
                  }//end dofZ
                }//end nodeX
              }//end nodeY
            }//end nodeZ
          }//end xi
        }//end yi
      }//end zi
    }//end i
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, loc, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, loc, &arr2d);
  } else {
    DMDAVecRestoreArrayDOF(da, loc, &arr3d);
  }

  VecZeroEntries(out);

  DMLocalToGlobalBegin(da, loc, ADD_VALUES, out);
  DMLocalToGlobalEnd(da, loc, ADD_VALUES, out);

  DMRestoreLocalVector(da, &loc);

  long double jac = hx * 0.5L;
  if(dim > 1) {
    jac *= (hy * 0.5L);
  }
  if(dim > 2) {
    jac *= (hz * 0.5L);
  }

  VecScale(out, jac);

  setBoundariesZero(da, out, K);
}

void computePstar(DM da, Vec vec, std::vector<int>& pStar) {
  pStar.clear();

  PetscInt dim;
  PetscInt px;
  PetscInt py;
  PetscInt pz;
  PetscInt dofsPerNode;
  DMDAGetInfo(da, &dim, PETSC_NULL, PETSC_NULL, PETSC_NULL, &px, &py, &pz,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

#ifdef DEBUG
  assert(nx >= 5);
  if(dim > 1) {
    assert(ny >= 5);
  }
  if(dim > 2) {
    assert(nz >= 5);
  }
#endif

  if(dim < 2) {
    ys = 0;
    ny = 1;
  }
  if(dim < 3) {
    zs = 0;
    nz = 1;
  }

  std::vector<PointAndVal> list;
  if(dim == 1) {
    PetscScalar** arr;
    DMDAVecGetArrayDOF(da, vec, &arr);
    for(int xi = xs; xi < (xs + nx); ++xi) {
      double val = 0.0;
      for(int d = 0; d < dofsPerNode; ++d) {
        double tmp = fabs(arr[xi][d]);
        if(tmp > val) {
          val = tmp;
        }
      }//end d
      if(val > 1.0e-12) {
        PointAndVal tmp;
        tmp.x = xi;
        tmp.y = 0;
        tmp.z = 0;
        tmp.v = val;
        list.push_back(tmp);
      }
    }//end xi
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr;
    DMDAVecGetArrayDOF(da, vec, &arr);
    for(int yi = ys; yi < (ys + ny); ++yi) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        double val = 0.0;
        for(int d = 0; d < dofsPerNode; ++d) {
          double tmp = fabs(arr[yi][xi][d]);
          if(tmp > val) {
            val = tmp;
          }
        }//end d
        if(val > 1.0e-12) {
          PointAndVal tmp;
          tmp.x = xi;
          tmp.y = yi;
          tmp.z = 0;
          tmp.v = val;
          list.push_back(tmp);
        }
      }//end xi
    }//end yi
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    PetscScalar**** arr;
    DMDAVecGetArrayDOF(da, vec, &arr);
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          double val = 0.0;
          for(int d = 0; d < dofsPerNode; ++d) {
            double tmp = fabs(arr[zi][yi][xi][d]);
            if(tmp > val) {
              val = tmp;
            }
          }//end d
          if(val > 1.0e-12) {
            PointAndVal tmp;
            tmp.x = xi;
            tmp.y = yi;
            tmp.z = zi;
            tmp.v = val;
            list.push_back(tmp);
          }
        }//end xi
      }//end yi
    }//end zi
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  }

  std::sort(list.begin(), list.end());

  std::vector<int> map((nx*ny*nz), -1);
  for(size_t i = 0; i < list.size(); ++i) {
    int dx = (list[i].x - xs);
    int dy = (list[i].y - ys);
    int dz = (list[i].z - zs);
    map[(((dz*ny)+ dy)*nx) + dx] = i;
  }//end i

  std::vector<double> vStar;
  for(int i = (((int)(list.size())) - 1); i >= 0; --i) {
    int x = list[i].x;
    int y = list[i].y;
    int z = list[i].z;
    int dx = x - xs;
    int dy = y - ys;
    int dz = z - zs;
    int idx = map[(((dz*ny) + dy)*nx) + dx];
    if(idx >= 0) {
      pStar.push_back(x);
      if(dim > 1) {
        pStar.push_back(y);
      }
      if(dim > 2) {
        pStar.push_back(z);
      }
      map[(((dz*ny) + dy)*nx) + dx] = vStar.size();
      vStar.push_back(list[i].v);
      for(int l = -2; l < 3; ++l) {
        if((x + l) < xs) {
          continue;
        }
        if((x + l) >= (xs + nx)) {
          continue;
        }
        for(int m = -2; m < 3; ++m) {
          if((y + m) < ys) {
            continue;
          }
          if((y + m) >= (ys + ny)) {
            continue;
          }
          for(int n = -2; n < 3; ++n) {
            if((z + n) < zs) {
              continue;
            }
            if((z + n) >= (zs + nz)) {
              continue;
            }
            if(l || m || n) {
              int ox = dx + l;
              int oy = dy + m;
              int oz = dz + n;
              map[(((oz*ny) + oy)*nx) + ox] = -1;
            }
          }//end n
        }//end m
      }//end l
    }
  }//end i

  list.clear();

  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)da), &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(dim == 1) {
    std::vector<double> sendVstar;
    std::vector<int> sendPstar;
    MPI_Request sReq1;
    int numSend = 0;
    if(rank > 0) {
      for(int dx = 0; dx < 2; ++dx) {
        int idx = map[dx];
        if(idx >= 0) {
          sendVstar.push_back(vStar[idx]);
          sendPstar.push_back(pStar[idx]);
        }
      }//end dx
      numSend = sendVstar.size();
      MPI_Isend(&numSend, 1, MPI_INT, (rank - 1), 1, comm, &sReq1);
    }
    MPI_Request sReq2;
    MPI_Request sReq3;
    if(numSend > 0) {
      MPI_Isend(&(sendVstar[0]), numSend, MPI_DOUBLE, (rank - 1), 2, comm, &sReq2);
      MPI_Isend(&(sendPstar[0]), numSend, MPI_INT, (rank - 1), 3, comm, &sReq3);
    }
    std::vector<double> recvVstar;
    std::vector<int> recvPstar;
    int numRecv = 0;
    if(rank < (px - 1)) {
      MPI_Status status;
      MPI_Recv(&numRecv, 1, MPI_INT, (rank + 1), 1, comm, &status);
      recvVstar.resize(numRecv);
      recvPstar.resize(numRecv);
    }
    if(numRecv > 0) {
      MPI_Status status;
      MPI_Request rReq2;
      MPI_Request rReq3;
      MPI_Irecv(&(recvVstar[0]), numRecv, MPI_DOUBLE, (rank + 1), 2, comm, &rReq2);
      MPI_Irecv(&(recvPstar[0]), numRecv, MPI_INT, (rank + 1), 3, comm, &rReq3);
      MPI_Wait(&rReq2, &status);
      MPI_Wait(&rReq3, &status);
    }
    std::vector<int> sendFlgs(recvVstar.size(), 0);
    for(size_t i = 0; i < recvVstar.size(); ++i) {
      for(int l = -2; l < 0; ++l) {
        int t = recvPstar[i] + l; 
        if((t >= xs) && (t < (xs + nx))) {
          int idx = map[t - xs];
          if(idx >= 0) {
            if(vStar[idx] > recvVstar[i]) {
              sendFlgs[i] = 1;
            } else {
              map[t - xs] = -1;
            }
          }//idx
        }//t
      }//end l
    }//end i
    MPI_Request sReq4;
    if(numRecv > 0) {
      MPI_Isend(&(sendFlgs[0]), numRecv, MPI_INT, (rank + 1), 4, comm, &sReq4);
    }
    std::vector<int> recvFlgs(sendVstar.size());
    if(numSend > 0) {
      MPI_Status status;
      MPI_Recv(&(recvFlgs[0]), numSend, MPI_INT, (rank - 1), 4, comm, &status);
    }
    for(size_t i = 0; i < recvFlgs.size(); ++i) {
      if(recvFlgs[i] == 1) {
        map[sendPstar[i] - xs] = -1;
      }
    }//end i
    if(rank > 0) {
      MPI_Status status;
      MPI_Wait(&sReq1, &status);
    }
    if(numSend > 0) {
      MPI_Status status;
      MPI_Wait(&sReq2, &status);
      MPI_Wait(&sReq3, &status);
    }
    if(numRecv > 0) {
      MPI_Status status;
      MPI_Wait(&sReq4, &status);
    }
  } else if(dim == 2) {
    int rj = rank/px;
    int ri = rank%px;
    std::vector<double> sendVstar1;
    std::vector<int> sendPstar1;
    MPI_Request sReq1;
    int numSend1 = 0;
    if(ri > 0) {
      for(int dy = 0; dy < ny; ++dy) {
        for(int dx = 0; dx < 2; ++dx) {
          int idx = map[(dy*nx) + dx];
          if(idx >= 0) {
            sendVstar1.push_back(vStar[idx]);
            sendPstar1.push_back(pStar[2*idx]);
            sendPstar1.push_back(pStar[(2*idx) + 1]);
          }
        }//end dx
      }//end dy
      numSend1 = sendVstar1.size();
      int other = (rj*px) + ri - 1;
      MPI_Isend(&numSend1, 1, MPI_INT, other, 1, comm, &sReq1);
    }
    MPI_Request sReq2;
    MPI_Request sReq3;
    if(numSend1 > 0) {
      int other = (rj*px) + ri - 1;
      MPI_Isend(&(sendVstar1[0]), numSend1, MPI_DOUBLE, other, 2, comm, &sReq2);
      MPI_Isend(&(sendPstar1[0]), (2*numSend1), MPI_INT, other, 3, comm, &sReq3);
    }
    std::vector<double> recvVstar1;
    std::vector<int> recvPstar1;
    int numRecv1 = 0;
    if(ri < (px - 1)) {
      MPI_Status status;
      int other = (rj*px) + ri + 1;
      MPI_Recv(&numRecv1, 1, MPI_INT, other, 1, comm, &status);
      recvVstar1.resize(numRecv1);
      recvPstar1.resize(2*numRecv1);
    }
    if(numRecv1 > 0) {
      MPI_Status status;
      MPI_Request rReq2;
      MPI_Request rReq3;
      int other = (rj*px) + ri + 1;
      MPI_Irecv(&(recvVstar1[0]), numRecv1, MPI_DOUBLE, other, 2, comm, &rReq2);
      MPI_Irecv(&(recvPstar1[0]), (2*numRecv1), MPI_INT, other, 3, comm, &rReq3);
      MPI_Wait(&rReq2, &status);
      MPI_Wait(&rReq3, &status);
    }    
    std::vector<int> sendFlgs1(recvVstar1.size(), 0);
    for(size_t i = 0; i < recvVstar1.size(); ++i) {
      for(int m = -2; m < 3; ++m) {
        for(int l = -2; l < 0; ++l) {
          int t1 = recvPstar1[2*i] + l; 
          int t2 = recvPstar1[(2*i) + 1] + m; 
          if((t1 >= xs) && (t1 < (xs + nx))) {
            int dx = (t1 - xs);
            if((t2 >= ys) && (t2 < (ys + ny))) {
              int dy = (t2 - ys);
              int idx = map[(dy*nx) + dx];
              if(idx >= 0) {
                if(vStar[idx] > recvVstar1[i]) {
                  sendFlgs1[i] = 1;
                } else {
                  map[(dy*nx) + dx] = -1;
                }
              }//idx
            }//t2
          }//t1
        }//end l
      }//end m
    }//end i
    std::vector<int> tmpMap;
    std::vector<double> sendVstar2;
    std::vector<int> sendPstar2;
    MPI_Request sReq4;
    int numSend2 = 0;
    if(rj > 0) {
      for(size_t i = 0; i < recvVstar1.size(); ++i) {
        if(sendFlgs1[i] == 0) {
          if((recvPstar1[(2*i) + 1] == ys) ||
              (recvPstar1[(2*i) + 1] == (ys + 1))) {
            tmpMap.push_back(i);
            sendVstar2.push_back(recvVstar1[i]);
            sendPstar2.push_back(recvPstar1[2*i]);
            sendPstar2.push_back(recvPstar1[(2*i) + 1]);
          }
        }
      }//end i
      for(int dy = 0; dy < 2; ++dy) {
        for(int dx = 0; dx < nx; ++dx) {
          int idx = map[(dy*nx) + dx];
          if(idx >= 0) {
            sendVstar2.push_back(vStar[idx]);
            sendPstar2.push_back(pStar[2*idx]);
            sendPstar2.push_back(pStar[(2*idx) + 1]);
          }
        }//end dx
      }//end dy
      numSend2 = sendVstar2.size();
      int other = ((rj - 1)*px) + ri;
      MPI_Isend(&numSend2, 1, MPI_INT, other, 4, comm, &sReq4);
    }
    MPI_Request sReq5;
    MPI_Request sReq6;
    if(numSend2 > 0) {
      int other = ((rj - 1)*px) + ri;
      MPI_Isend(&(sendVstar2[0]), numSend2, MPI_DOUBLE, other, 5, comm, &sReq5);
      MPI_Isend(&(sendPstar2[0]), (2*numSend2), MPI_INT, other, 6, comm, &sReq6);
    }
    std::vector<double> recvVstar2;
    std::vector<int> recvPstar2;
    int numRecv2 = 0;
    if(rj < (py - 1)) {
      MPI_Status status;
      int other = ((rj + 1)*px) + ri;
      MPI_Recv(&numRecv2, 1, MPI_INT, other, 4, comm, &status);
      recvVstar2.resize(numRecv2);
      recvPstar2.resize(2*numRecv2);
    }
    if(numRecv2 > 0) {
      MPI_Status status;
      MPI_Request rReq5;
      MPI_Request rReq6;
      int other = ((rj + 1)*px) + ri;
      MPI_Irecv(&(recvVstar2[0]), numRecv2, MPI_DOUBLE, other, 5, comm, &rReq5);
      MPI_Irecv(&(recvPstar2[0]), (2*numRecv2), MPI_INT, other, 6, comm, &rReq6);
      MPI_Wait(&rReq5, &status);
      MPI_Wait(&rReq6, &status);
    }    
    std::vector<int> sendFlgs2(recvVstar2.size(), 0);
    for(size_t i = 0; i < recvVstar2.size(); ++i) {
      for(int m = -2; m < 0; ++m) {
        for(int l = -2; l < 3; ++l) {
          int t1 = recvPstar2[2*i] + l; 
          int t2 = recvPstar2[(2*i) + 1] + m; 
          if((t1 >= xs) && (t1 < (xs + nx))) {
            int dx = (t1 - xs);
            if((t2 >= ys) && (t2 < (ys + ny))) {
              int dy = (t2 - ys);
              int idx = map[(dy*nx) + dx];
              if(idx >= 0) {
                if(vStar[idx] > recvVstar2[i]) {
                  sendFlgs2[i] = 1;
                } else {
                  map[(dy*nx) + dx] = -1;
                }
              }//idx
            }//t2
          }//t1
        }//end l
      }//end m
    }//end i
    MPI_Request sReq7;
    if(numRecv2 > 0) {
      int other = ((rj + 1)*px) + ri;
      MPI_Isend(&(sendFlgs2[0]), numRecv2, MPI_INT, other, 7, comm, &sReq7);
    }
    std::vector<int> recvFlgs2(sendVstar2.size());
    if(numSend2 > 0) {
      MPI_Status status;
      int other = ((rj - 1)*px) + ri;
      MPI_Recv(&(recvFlgs2[0]), numSend2, MPI_INT, other, 7, comm, &status);
    }
    for(size_t i = 0; i < tmpMap.size(); ++i) {
      if(recvFlgs2[i] == 1) {
        sendFlgs1[tmpMap[i]] = 1;
      }
    }//end i
    for(size_t i = tmpMap.size(); i < recvFlgs2.size(); ++i) {
      if(recvFlgs2[i] == 1) {
        int dx = sendPstar2[2*i] - xs;
        int dy = sendPstar2[(2*i) + 1] - ys;
        map[(dy*nx) + dx] = -1;
      }
    }
    MPI_Request sReq8;
    if(numRecv1 > 0) {
      int other = (rj*px) + ri + 1;
      MPI_Isend(&(sendFlgs1[0]), numRecv1, MPI_INT, other, 8, comm, &sReq8);
    }
    std::vector<int> recvFlgs1(sendVstar1.size());
    if(numSend1 > 0) {
      MPI_Status status;
      int other = (rj*px) + ri - 1;
      MPI_Recv(&(recvFlgs1[0]), numSend1, MPI_INT, other, 8, comm, &status);
    }
    for(size_t i = 0; i < recvFlgs1.size(); ++i) {
      if(recvFlgs1[i] == 1) {
        int dx = sendPstar1[2*i] - xs;
        int dy = sendPstar1[(2*i) + 1] - ys;
        map[(dy*nx) + dx] = -1;
      }
    }//end i
    if(ri > 0) {
      MPI_Status status;
      MPI_Wait(&sReq1, &status);
    }
    if(numSend1 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq2, &status);
      MPI_Wait(&sReq3, &status);
    }
    if(rj > 0) {
      MPI_Status status;
      MPI_Wait(&sReq4, &status);
    }
    if(numSend2 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq5, &status);
      MPI_Wait(&sReq6, &status);
    }
    if(numRecv2 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq7, &status);
    }
    if(numRecv1 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq8, &status);
    }
  } else {
    int rk = rank/(px*py);
    int rj = (rank/px)%py;
    int ri = rank%px;
    std::vector<double> sendVstar1;
    std::vector<int> sendPstar1;
    MPI_Request sReq1;
    int numSend1 = 0;
    if(ri > 0) {
      for(int dz = 0; dz < nz; ++dz) {
        for(int dy = 0; dy < ny; ++dy) {
          for(int dx = 0; dx < 2; ++dx) {
            int idx = map[(((dz*ny) + dy)*nx) + dx];
            if(idx >= 0) {
              sendVstar1.push_back(vStar[idx]);
              sendPstar1.push_back(pStar[3*idx]);
              sendPstar1.push_back(pStar[(3*idx) + 1]);
              sendPstar1.push_back(pStar[(3*idx) + 2]);
            }
          }//end dx
        }//end dy
      }//end dz
      numSend1 = sendVstar1.size();
      int other = (((rk*py) + rj)*px) + ri - 1;
      MPI_Isend(&numSend1, 1, MPI_INT, other, 1, comm, &sReq1);
    }
    MPI_Request sReq2;
    MPI_Request sReq3;
    if(numSend1 > 0) {
      int other = (((rk*py) + rj)*px) + ri - 1;
      MPI_Isend(&(sendVstar1[0]), numSend1, MPI_DOUBLE, other, 2, comm, &sReq2);
      MPI_Isend(&(sendPstar1[0]), (3*numSend1), MPI_INT, other, 3, comm, &sReq3);
    }
    std::vector<double> recvVstar1;
    std::vector<int> recvPstar1;
    int numRecv1 = 0;
    if(ri < (px - 1)) {
      MPI_Status status;
      int other = (((rk*py) + rj)*px) + ri + 1;
      MPI_Recv(&numRecv1, 1, MPI_INT, other, 1, comm, &status);
      recvVstar1.resize(numRecv1);
      recvPstar1.resize(3*numRecv1);
    }
    if(numRecv1 > 0) {
      MPI_Status status;
      MPI_Request rReq2;
      MPI_Request rReq3;
      int other = (((rk*py) + rj)*px) + ri + 1;
      MPI_Irecv(&(recvVstar1[0]), numRecv1, MPI_DOUBLE, other, 2, comm, &rReq2);
      MPI_Irecv(&(recvPstar1[0]), (3*numRecv1), MPI_INT, other, 3, comm, &rReq3);
      MPI_Wait(&rReq2, &status);
      MPI_Wait(&rReq3, &status);
    }    
    std::vector<int> sendFlgs1(recvVstar1.size(), 0);
    for(size_t i = 0; i < recvVstar1.size(); ++i) {
      for(int n = -2; n < 3; ++n) {
        for(int m = -2; m < 3; ++m) {
          for(int l = -2; l < 0; ++l) {
            int t1 = recvPstar1[3*i] + l; 
            int t2 = recvPstar1[(3*i) + 1] + m; 
            int t3 = recvPstar1[(3*i) + 2] + n; 
            if((t1 >= xs) && (t1 < (xs + nx))) {
              int dx = (t1 - xs);
              if((t2 >= ys) && (t2 < (ys + ny))) {
                int dy = (t2 - ys);
                if((t3 >= zs) && (t3 < (zs + nz))) {
                  int dz = (t3 - zs);
                  int idx = map[(((dz*ny) + dy)*nx) + dx];
                  if(idx >= 0) {
                    if(vStar[idx] > recvVstar1[i]) {
                      sendFlgs1[i] = 1;
                    } else {
                      map[(((dz*ny) + dy)*nx) + dx] = -1;
                    }
                  }//idx
                }//t3
              }//t2
            }//t1
          }//end l
        }//end m
      }//end n
    }//end i
    std::vector<int> tmpMap1;
    std::vector<double> sendVstar2;
    std::vector<int> sendPstar2;
    MPI_Request sReq4;
    int numSend2 = 0;
    if(rj > 0) {
      for(size_t i = 0; i < recvVstar1.size(); ++i) {
        if(sendFlgs1[i] == 0) {
          if((recvPstar1[(3*i) + 1] == ys) ||
              (recvPstar1[(3*i) + 1] == (ys + 1))) {
            tmpMap1.push_back(i);
            sendVstar2.push_back(recvVstar1[i]);
            sendPstar2.push_back(recvPstar1[3*i]);
            sendPstar2.push_back(recvPstar1[(3*i) + 1]);
            sendPstar2.push_back(recvPstar1[(3*i) + 2]);
          }
        }
      }//end i
      for(int dz = 0; dz < nz; ++dz) {
        for(int dy = 0; dy < 2; ++dy) {
          for(int dx = 0; dx < nx; ++dx) {
            int idx = map[(((dz*ny) + dy)*nx) + dx];
            if(idx >= 0) {
              sendVstar2.push_back(vStar[idx]);
              sendPstar2.push_back(pStar[3*idx]);
              sendPstar2.push_back(pStar[(3*idx) + 1]);
              sendPstar2.push_back(pStar[(3*idx) + 2]);
            }
          }//end dx
        }//end dy
      }//end dz
      numSend2 = sendVstar2.size();
      int other = (((rk*py) + (rj - 1))*px) + ri;
      MPI_Isend(&numSend2, 1, MPI_INT, other, 4, comm, &sReq4);
    }
    MPI_Request sReq5;
    MPI_Request sReq6;
    if(numSend2 > 0) {
      int other = (((rk*py) + (rj - 1))*px) + ri;
      MPI_Isend(&(sendVstar2[0]), numSend2, MPI_DOUBLE, other, 5, comm, &sReq5);
      MPI_Isend(&(sendPstar2[0]), (3*numSend2), MPI_INT, other, 6, comm, &sReq6);
    }
    std::vector<double> recvVstar2;
    std::vector<int> recvPstar2;
    int numRecv2 = 0;
    if(rj < (py - 1)) {
      MPI_Status status;
      int other = (((rk*py) + (rj + 1))*px) + ri;
      MPI_Recv(&numRecv2, 1, MPI_INT, other, 4, comm, &status);
      recvVstar2.resize(numRecv2);
      recvPstar2.resize(3*numRecv2);
    }
    if(numRecv2 > 0) {
      MPI_Status status;
      MPI_Request rReq5;
      MPI_Request rReq6;
      int other = (((rk*py) + (rj + 1))*px) + ri;
      MPI_Irecv(&(recvVstar2[0]), numRecv2, MPI_DOUBLE, other, 5, comm, &rReq5);
      MPI_Irecv(&(recvPstar2[0]), (3*numRecv2), MPI_INT, other, 6, comm, &rReq6);
      MPI_Wait(&rReq5, &status);
      MPI_Wait(&rReq6, &status);
    }    
    std::vector<int> sendFlgs2(recvVstar2.size(), 0);
    for(size_t i = 0; i < recvVstar2.size(); ++i) {
      for(int n = -2; n < 3; ++n) {
        for(int m = -2; m < 0; ++m) {
          for(int l = -2; l < 3; ++l) {
            int t1 = recvPstar2[3*i] + l; 
            int t2 = recvPstar2[(3*i) + 1] + m; 
            int t3 = recvPstar2[(3*i) + 2] + n; 
            if((t1 >= xs) && (t1 < (xs + nx))) {
              int dx = (t1 - xs);
              if((t2 >= ys) && (t2 < (ys + ny))) {
                int dy = (t2 - ys);
                if((t3 >= zs) && (t3 < (zs + nz))) {
                  int dz = (t3 - zs);
                  int idx = map[(((dz*ny) + dy)*nx) + dx];
                  if(idx >= 0) {
                    if(vStar[idx] > recvVstar2[i]) {
                      sendFlgs2[i] = 1;
                    } else {
                      map[(((dz*ny) + dy)*nx) + dx] = -1;
                    }
                  }//idx
                }//t3
              }//t2
            }//t1
          }//end l
        }//end m
      }//end n
    }//end i
    std::vector<int> tmpMap2;
    std::vector<double> sendVstar3;
    std::vector<int> sendPstar3;
    MPI_Request sReq7;
    int numSend3 = 0;
    if(rk > 0) {
      for(size_t i = 0; i < recvVstar2.size(); ++i) {
        if(sendFlgs2[i] == 0) {
          if((recvPstar2[(3*i) + 2] == zs) ||
              (recvPstar2[(3*i) + 2] == (zs + 1))) {
            tmpMap2.push_back(i);
            sendVstar3.push_back(recvVstar2[i]);
            sendPstar3.push_back(recvPstar2[3*i]);
            sendPstar3.push_back(recvPstar2[(3*i) + 1]);
            sendPstar3.push_back(recvPstar2[(3*i) + 2]);
          }
        }
      }//end i
      for(int dz = 0; dz < 2; ++dz) {
        for(int dy = 0; dy < ny; ++dy) {
          for(int dx = 0; dx < nx; ++dx) {
            int idx = map[(((dz*ny) + dy)*nx) + dx];
            if(idx >= 0) {
              sendVstar3.push_back(vStar[idx]);
              sendPstar3.push_back(pStar[3*idx]);
              sendPstar3.push_back(pStar[(3*idx) + 1]);
              sendPstar3.push_back(pStar[(3*idx) + 2]);
            }
          }//end dx
        }//end dy
      }//end dz
      numSend3 = sendVstar3.size();
      int other = ((((rk - 1)*py) + rj)*px) + ri;
      MPI_Isend(&numSend3, 1, MPI_INT, other, 7, comm, &sReq7);
    }
    MPI_Request sReq8;
    MPI_Request sReq9;
    if(numSend3 > 0) {
      int other = ((((rk - 1)*py) + rj)*px) + ri;
      MPI_Isend(&(sendVstar3[0]), numSend3, MPI_DOUBLE, other, 8, comm, &sReq8);
      MPI_Isend(&(sendPstar3[0]), (3*numSend3), MPI_INT, other, 9, comm, &sReq9);
    }
    std::vector<double> recvVstar3;
    std::vector<int> recvPstar3;
    int numRecv3 = 0;
    if(rk < (pz - 1)) {
      MPI_Status status;
      int other = ((((rk + 1)*py) + rj)*px) + ri;
      MPI_Recv(&numRecv3, 1, MPI_INT, other, 7, comm, &status);
      recvVstar3.resize(numRecv3);
      recvPstar3.resize(3*numRecv3);
    }
    if(numRecv3 > 0) {
      MPI_Status status;
      MPI_Request rReq8;
      MPI_Request rReq9;
      int other = ((((rk + 1)*py) + rj)*px) + ri;
      MPI_Irecv(&(recvVstar3[0]), numRecv3, MPI_DOUBLE, other, 8, comm, &rReq8);
      MPI_Irecv(&(recvPstar3[0]), (3*numRecv3), MPI_INT, other, 9, comm, &rReq9);
      MPI_Wait(&rReq8, &status);
      MPI_Wait(&rReq9, &status);
    }    
    std::vector<int> sendFlgs3(recvVstar3.size(), 0);
    for(size_t i = 0; i < recvVstar3.size(); ++i) {
      for(int n = -2; n < 0; ++n) {
        for(int m = -2; m < 3; ++m) {
          for(int l = -2; l < 3; ++l) {
            int t1 = recvPstar3[3*i] + l; 
            int t2 = recvPstar3[(3*i) + 1] + m; 
            int t3 = recvPstar3[(3*i) + 2] + n; 
            if((t1 >= xs) && (t1 < (xs + nx))) {
              int dx = (t1 - xs);
              if((t2 >= ys) && (t2 < (ys + ny))) {
                int dy = (t2 - ys);
                if((t3 >= zs) && (t3 < (zs + nz))) {
                  int dz = (t3 - zs);
                  int idx = map[(((dz*ny) + dy)*nx) + dx];
                  if(idx >= 0) {
                    if(vStar[idx] > recvVstar3[i]) {
                      sendFlgs3[i] = 1;
                    } else {
                      map[(((dz*ny) + dy)*nx) + dx] = -1;
                    }
                  }//idx
                }//t3
              }//t2
            }//t1
          }//end l
        }//end m
      }//end n
    }//end i
    MPI_Request sReq10;
    if(numRecv3 > 0) {
      int other = ((((rk + 1)*py) + rj)*px) + ri;
      MPI_Isend(&(sendFlgs3[0]), numRecv3, MPI_INT, other, 10, comm, &sReq10);
    }
    std::vector<int> recvFlgs3(sendVstar3.size());
    if(numSend3 > 0) {
      MPI_Status status;
      int other = ((((rk - 1)*py) + rj)*px) + ri;
      MPI_Recv(&(recvFlgs3[0]), numSend3, MPI_INT, other, 10, comm, &status);
    }
    for(size_t i = 0; i < tmpMap2.size(); ++i) {
      if(recvFlgs3[i] == 1) {
        sendFlgs2[tmpMap2[i]] = 1;
      }
    }//end i
    for(size_t i = tmpMap2.size(); i < recvFlgs3.size(); ++i) {
      if(recvFlgs3[i] == 1) {
        int dx = sendPstar3[3*i] - xs;
        int dy = sendPstar3[(3*i) + 1] - ys;
        int dz = sendPstar3[(3*i) + 2] - zs;
        map[(((dz*ny) + dy)*nx) + dx] = -1;
      }
    }//end i
    MPI_Request sReq11;
    if(numRecv2 > 0) {
      int other = (((rk*py) + (rj + 1))*px) + ri;
      MPI_Isend(&(sendFlgs2[0]), numRecv2, MPI_INT, other, 11, comm, &sReq11);
    }
    std::vector<int> recvFlgs2(sendVstar2.size());
    if(numSend2 > 0) {
      MPI_Status status;
      int other = (((rk*py) + (rj - 1))*px) + ri;
      MPI_Recv(&(recvFlgs2[0]), numSend2, MPI_INT, other, 11, comm, &status);
    }
    for(size_t i = 0; i < tmpMap1.size(); ++i) {
      if(recvFlgs2[i] == 1) {
        sendFlgs1[tmpMap1[i]] = 1;
      }
    }//end i
    for(size_t i = tmpMap1.size(); i < recvFlgs2.size(); ++i) {
      if(recvFlgs2[i] == 1) {
        int dx = sendPstar2[3*i] - xs;
        int dy = sendPstar2[(3*i) + 1] - ys;
        int dz = sendPstar2[(3*i) + 2] - zs;
        map[(((dz*ny) + dy)*nx) + dx] = -1;
      }
    }//end i
    MPI_Request sReq12;
    if(numRecv1 > 0) {
      int other = (((rk*py) + rj)*px) + ri + 1;
      MPI_Isend(&(sendFlgs1[0]), numRecv1, MPI_INT, other, 12, comm, &sReq12);
    }
    std::vector<int> recvFlgs1(sendVstar1.size());
    if(numSend1 > 0) {
      MPI_Status status;
      int other = (((rk*py) + rj)*px) + ri - 1;
      MPI_Recv(&(recvFlgs1[0]), numSend1, MPI_INT, other, 12, comm, &status);
    }
    for(size_t i = 0; i < recvFlgs1.size(); ++i) {
      if(recvFlgs1[i] == 1) {
        int dx = sendPstar1[3*i] - xs;
        int dy = sendPstar1[(3*i) + 1] - ys;
        int dz = sendPstar1[(3*i) + 2] - zs;
        map[(((dz*ny) + dy)*nx) + dx] = -1;
      }
    }//end i
    if(ri > 0) {
      MPI_Status status;
      MPI_Wait(&sReq1, &status);
    }
    if(numSend1 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq2, &status);
      MPI_Wait(&sReq3, &status);
    }
    if(rj > 0) {
      MPI_Status status;
      MPI_Wait(&sReq4, &status);
    }
    if(numSend2 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq5, &status);
      MPI_Wait(&sReq6, &status);
    }
    if(rk > 0) {
      MPI_Status status;
      MPI_Wait(&sReq7, &status);
    }
    if(numSend3 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq8, &status);
      MPI_Wait(&sReq9, &status);
    }
    if(numRecv3 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq10, &status);
    }
    if(numRecv2 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq11, &status);
    }
    if(numRecv1 > 0) {
      MPI_Status status;
      MPI_Wait(&sReq12, &status);
    }
  }

  pStar.clear();
  vStar.clear();

  if(dim == 1) {
    for(int xi = xs, cnt = 0; xi < (xs + nx); ++xi, ++cnt) {
      if(map[cnt] >= 0) {
        pStar.push_back(xi);
      }
    }//end xi
  } else if(dim == 2) {
    for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
      for(int xi = xs; xi < (xs + nx); ++xi, ++cnt) {
        if(map[cnt] >= 0) {
          pStar.push_back(xi);
          pStar.push_back(yi);
        }
      }//end xi
    }//end yi
  } else {
    for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi, ++cnt) {
          if(map[cnt] >= 0) {
            pStar.push_back(xi);
            pStar.push_back(yi);
            pStar.push_back(zi);
          }
        }//end xi
      }//end yi
    }//end zi
  }
}


