
#include "gmg/include/loa.h"
#include <algorithm>

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
  PetscInt px;
  PetscInt py;
  PetscInt pz;
  PetscInt dofsPerNode;
  DMDAGetInfo(data->daH, &dim, &Nx, &Ny, &Nz, &px, &py, &pz,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(data->daH, &xs, &ys, &zs, &nx, &ny, &nz);

#ifdef DEBUG
  assert(nx >= 5);
  if(dim > 1) {
    assert(ny >= 5);
  }
  if(dim > 2) {
    assert(nz >= 5);
  }
#endif

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy;
  if(dim < 2) {
    ys = 0;
    ny = 1;
    Ny = 1;
  } else {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz;
  if(dim < 3) {
    zs = 0;
    nz = 1;
    Nz = 1;
  } else {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  std::vector<PointAndVal> list;
  if(dim == 1) {
    PetscScalar** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
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
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
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
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  } else {
    PetscScalar**** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
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
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  }

  std::sort(list.begin(), list.end());

  std::vector<int> map((nx*ny*nz), -1);
  for(int i = 0; i < list.size(); ++i) {
    int dx = (list[i].x - xs);
    int dy = (list[i].y - ys);
    int dz = (list[i].z - zs);
    map[(((dz*ny)+ dy)*nx) + dx] = i;
  }//end i

  std::vector<int> pStar;
  std::vector<double> vStar;
  for(int i = (list.size() - 1); i >= 0; --i) {
    if(list[i].v > 1.0e-12) {
      int x = list[i].x;
      int y = list[i].y;
      int z = list[i].z;
      pStar.push_back(x);
      if(dim > 1) {
        pStar.push_back(y);
      }
      if(dim > 2) {
        pStar.push_back(z);
      }
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
              int ox = x + l - xs;
              int oy = y + m - ys;
              int oz = z + n - zs;
              int idx = map[(((oz*ny) + oy)*nx) + ox];
              if(idx < i) {
                list[idx].v = 0; 
              }
            }
          }//end n
        }//end m
      }//end l
    }
  }//end i

  list.clear();

  for(int i = 0; i < map.size(); ++i) {
    map[i] = -1;
  }//end i
  for(int i = 0; i < vStar.size(); ++i) {
    if(dim == 1) {
      int dx = (pStar[i] - xs);
      map[dx] = i;
    } else if(dim == 2) {
      int dx = (pStar[2*i] - xs);
      int dy = (pStar[(2*i) + 1] - ys);
      map[(dy*nx) + dx] = i;
    } else {
      int dx = (pStar[3*i] - xs);
      int dy = (pStar[(3*i) + 1] - ys);
      int dz = (pStar[(3*i) + 2] - zs);
      map[(((dz*ny)+ dy)*nx) + dx] = i;
    }
  }//end i

  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)(data->daH)), &comm);

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
    for(int i = 0; i < recvVstar.size(); ++i) {
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
          }
        }
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
    for(int i = 0; i < recvFlgs.size(); ++i) {
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
    for(int i = 0; i < recvVstar1.size(); ++i) {
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
              }
            }
          }
        }//end l
      }//end m
    }//end i
    std::vector<int> tmpMap;
    std::vector<double> sendVstar2;
    std::vector<int> sendPstar2;
    MPI_Request sReq4;
    int numSend2 = 0;
    if(rj > 0) {
      for(int i = 0; i < recvVstar1.size(); ++i) {
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
    for(int i = 0; i < recvVstar2.size(); ++i) {
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
              }
            }
          }
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
    for(int i = 0; i < tmpMap.size(); ++i) {
      if(recvFlgs2[i] == 1) {
        sendFlgs1[tmpMap[i]] = 1;
      }
    }//end i
    for(int i = tmpMap.size(); i < recvFlgs2.size(); ++i) {
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
    for(int i = 0; i < recvFlgs1.size(); ++i) {
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
  }

}

/*
   void applyLS(LSdata* data, Vec g, Vec v1, Vec v2, double a[2],
   int maxIters, double tgtNorm, double currNorm) {
   MatMult((data->Kmat), v1, (data->w1));
   MatMult((data->Kmat), v2, (data->w2));
   double Hmat[2][2];
   VecDot((data->w1), (data->w1), &(Hmat[0][0]));
   VecDot((data->w1), (data->w2), &(Hmat[0][1]));
   Hmat[1][0] = Hmat[0][1];
   VecDot((data->w2), (data->w2), &(Hmat[1][1]));
   double eig[2];
   eigenVals2x2(Hmat, eig);
   double minEig = ((eig[0] < eig[1]) ? (eig[0]) : (eig[1]));
   double Hinv[2][2];
   if(minEig <= 1.0e-12) {
   double shift = 1.0 - minEig;
   double Lmat[2][2];
   Lmat[0][0] = Hmat[0][0] + shift;
   Lmat[0][1] = Hmat[0][1];
   Lmat[1][0] = Hmat[1][0];
   Lmat[1][1] = Hmat[1][1] + shift;
   matInvert2x2(Lmat, Hinv);
   } else {
   matInvert2x2(Hmat, Hinv);
   }
   double w1g;
   double w2g;
   VecDot((data->w1), g, &w1g);
   VecDot((data->w2), g, &w2g);
   a[0] = a[1] = 0.0;
   double gDotG = currNorm*currNorm; 
   double tgtNrmSqr = tgtNorm*tgtNorm;
   double obj = gDotG;
   for(int iter = 0; iter < maxIters; ++iter) {
   if(obj <= tgtNrmSqr) {
   break;
   }
   double grad[2];
   grad[0] = -w1g + (a[0]*Hmat[0][0]) + (a[1]*Hmat[1][0]);
   grad[1] = -w2g + (a[0]*Hmat[0][1]) + (a[1]*Hmat[1][1]);
   if((fabs(grad[0]) <= 1.0e-12) && (fabs(grad[1]) <= 1.0e-12)) {
   break;
   }
   double step[2];
   matMult2x2(Hinv, grad, step);
   if((fabs(step[0]) <= 1.0e-12) && (fabs(step[1]) <= 1.0e-12)) {
   break;
   }
   double alpha = 1.0;
   while(alpha >= 1.0e-12) {
   double tmp[2]; 
   tmp[0] = a[0] - (alpha*step[0]);
   tmp[1] = a[1] - (alpha*step[1]);
   double tmpObj = gDotG -2.0*((tmp[0]*w1g) + (tmp[1]*w2g));
   for(int r = 0; r < 2; ++r) {
   for(int c = 0; c < 2; ++c) {
   tmpObj += (tmp[r]*tmp[c]*Hmat[r][c]);
   }//end c
   }//end r
   if(tmpObj < obj) {
   obj = tmpObj;
   a[0] = tmp[0];
   a[1] = tmp[1];
   break;
   } else {
   alpha *= 0.5;
   }
   }
   if(alpha < 1.0e-12) {
   break;
   }
}//end iter
}
*/


