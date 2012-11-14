
#include <iostream>
#include <cmath>
#include <algorithm>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

extern PetscLogEvent createDAevent;

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy, 
    std::vector<std::vector<int> >& scanLx, MPI_Comm globalComm, bool print) {
  PetscLogEventBegin(createDAevent, 0, 0, 0, 0);

  createGridSizes(dim, Nz, Ny, Nx, print);

  int globalRank;
  int globalNpes;
  MPI_Comm_rank(globalComm, &globalRank);
  MPI_Comm_size(globalComm, &globalNpes);

  int maxCoarseNpes = globalNpes;
  PetscOptionsGetInt(PETSC_NULL, "-maxCoarseNpes", &maxCoarseNpes, PETSC_NULL);
  if(maxCoarseNpes > globalNpes) {
    maxCoarseNpes = globalNpes;
  }
#ifdef DEBUG
  assert(maxCoarseNpes > 0);
#endif

  int numLevels = Nx.size();
#ifdef DEBUG
  assert(numLevels > 0);
#endif
  activeNpes.resize(numLevels);
  activeComms.resize(numLevels);
  da.resize(numLevels);
  partZ.resize(numLevels);
  partY.resize(numLevels);
  partX.resize(numLevels);
  offsets.resize(numLevels);
  scanLz.resize(numLevels);
  scanLy.resize(numLevels);
  scanLx.resize(numLevels);

  MPI_Group globalGroup;
  MPI_Comm_group(globalComm, &globalGroup);

  int* rankList = new int[globalNpes];
  for(int i = 0; i < globalNpes; ++i) {
    rankList[i] = i;
  }//end for i

  //0 is the coarsest level.
  for(int lev = 0; lev < numLevels; ++lev) {
    int maxNpes;
    if(lev == 0) {
      maxNpes = maxCoarseNpes;
    } else {
      maxNpes = globalNpes;
    }
    computePartition(dim, Nz[lev], Ny[lev], Nx[lev], maxNpes, partZ[lev], partY[lev], partX[lev],
        offsets[lev], scanLz[lev], scanLy[lev], scanLx[lev]);
    PetscInt pz = (partZ[lev]).size();
    PetscInt py = (partY[lev]).size();
    PetscInt px = (partX[lev]).size();
    activeNpes[lev] = (px*py*pz);
    if(print) {
      std::cout<<"Active Npes for Level "<<lev<<" = "<<(activeNpes[lev])
        <<" : (px, py, pz) = ("<<px<<", "<<py<<", "<<pz<<")"<<std::endl;
    }
#ifdef DEBUG
    if(lev > 0) {
      assert(activeNpes[lev] >= activeNpes[lev - 1]);
    }
#endif
    if(globalRank < (activeNpes[lev])) {
      MPI_Group subGroup;
      MPI_Group_incl(globalGroup, (activeNpes[lev]), rankList, &subGroup);
      MPI_Comm_create(globalComm, subGroup, &(activeComms[lev]));
      MPI_Group_free(&subGroup);
      DACreate(activeComms[lev], dim, DA_NONPERIODIC, DA_STENCIL_BOX, (Nx[lev]), (Ny[lev]), (Nz[lev]),
          px, py, pz, dofsPerNode, 1, &(partX[lev][0]), &(partY[lev][0]), &(partZ[lev][0]), (&(da[lev])));
    } else {
      MPI_Comm_create(globalComm, MPI_GROUP_EMPTY, &(activeComms[lev]));
#ifdef DEBUG
      assert(activeComms[lev] == MPI_COMM_NULL);
#endif
      da[lev] = NULL;
    }
  }//end lev

  delete [] rankList;
  MPI_Group_free(&globalGroup);

  PetscLogEventEnd(createDAevent, 0, 0, 0, 0);
}

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> &lz, std::vector<PetscInt> &ly, std::vector<PetscInt> &lx,
    std::vector<int>& offsets, std::vector<int>& scanLz, std::vector<int>& scanLy, std::vector<int>& scanLx) {
#ifdef DEBUG
  if(dim < 3) {
    assert(Nz == 1);
  }
  if(dim < 2) {
    assert(Ny == 1);
  }
  assert(Nx > 0);
  assert(Ny > 0);
  assert(Nz > 0);
  assert(maxNpes > 0);
#endif

  std::vector<PetscInt> Nlist;
  Nlist.push_back(Nx);
  Nlist.push_back(Ny);
  Nlist.push_back(Nz);

  std::sort(Nlist.begin(), Nlist.end());

  double tmp = std::pow(((static_cast<double>(Nx*Ny*Nz))/(static_cast<double>(maxNpes))), (1.0/(static_cast<double>(dim))));

  std::vector<int> pList(3, 1);
  for(int d = 0; d < 3; ++d) {
    if(Nlist[d] > 1) {
      pList[d] = static_cast<int>(std::floor((static_cast<double>(Nlist[d]))/tmp));
      if(pList[d] > Nlist[d]) {
        pList[d] = Nlist[d];
      }
      if(pList[d] < 1) {
        pList[d] = 1;
      }
    }
  }//end d
#ifdef DEBUG
  assert(((pList[0])*(pList[1])*(pList[2])) <= maxNpes);
#endif

  bool partChanged;
  do {
    partChanged = false;
    for(int d = 2; d >= 0; --d) {
      if( pList[d] < Nlist[d] ) {
        if( ((pList[d] + 1)*(pList[(d+1)%3])*(pList[(d+2)%3])) <= maxNpes ) {
          ++(pList[d]);
          partChanged = true;
        }
      }
    }//end d
  } while(partChanged);

  int px;
  for(int d = 0; d < 3; ++d) {
    if(Nx == Nlist[d]) {
      px = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

  int py;
  for(int d = 0; d < 2; ++d) {
    if(Ny == Nlist[d]) {
      py = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

#ifdef DEBUG
  assert(Nz == Nlist[0]);
#endif

  int pz;
  pz = pList[0];

#ifdef DEBUG
  assert((px*py*pz) <= maxNpes);
  assert(px >= 1);
  assert(py >= 1);
  assert(pz >= 1);
  assert(px <= Nx);
  assert(py <= Ny);
  assert(pz <= Nz);
#endif

  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  lx.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(lx[cnt]);
  }//end cnt

  PetscInt avgY = Ny/py;
  PetscInt extraY = Ny%py; 
  ly.resize(py, avgY);
  for(int cnt = 0; cnt < extraY; ++cnt) {
    ++(ly[cnt]);
  }//end cnt

  PetscInt avgZ = Nz/pz;
  PetscInt extraZ = Nz%pz; 
  lz.resize(pz, avgZ);
  for(int cnt = 0; cnt < extraZ; ++cnt) {
    ++(lz[cnt]);
  }//end cnt

  int npes = px*py*pz;

  offsets.resize(npes);
  offsets[0] = 0;
  for(int p = 1; p < npes; ++p) {
    int k = (p - 1)/(px*py);
    int j = ((p - 1)/px)%py;
    int i = (p - 1)%px;
    offsets[p] = offsets[p - 1] + (lz[k]*ly[j]*lx[i]);
  }//end p

  scanLx.resize(px);
  scanLx[0] = lx[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanLx[i] = scanLx[i - 1] + lx[i];
  }//end i

  scanLy.resize(py);
  scanLy[0] = ly[0] - 1;
  for(int i = 1; i < py; ++i) {
    scanLy[i] = scanLy[i - 1] + ly[i];
  }//end i

  scanLz.resize(pz);
  scanLz[0] = lz[0] - 1;
  for(int i = 1; i < pz; ++i) {
    scanLz[i] = scanLz[i - 1] + lz[i];
  }//end i
}

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, bool print) {
#ifdef DEBUG
  assert(dim > 0);
  assert(dim <= 3);
#endif

  PetscInt currNx = 17;
  PetscInt currNy = 1;
  PetscInt currNz = 1;

  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  if(print) {
    std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
  }
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNy", &currNy, PETSC_NULL);
    if(print) {
      std::cout<<"Ny (Finest) = "<<currNy<<std::endl;
    }
  }
  if(dim > 2) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNz", &currNz, PETSC_NULL);
    if(print) {
      std::cout<<"Nz (Finest) = "<<currNz<<std::endl;
    }
  }

  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  if(print) {
    std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;
  }

  const unsigned int minGridSize = 9;

  Nx.clear();
  Ny.clear();
  Nz.clear();

  //0 is the coarsest level.
  for(int lev = 0; lev < maxNumLevels; ++lev) {
    Nx.insert(Nx.begin(), currNx);
    if(dim > 1) {
      Ny.insert(Ny.begin(), currNy);
    }
    if(dim > 2) {
      Nz.insert(Nz.begin(), currNz);
    }
    if( (currNx < minGridSize) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    if(dim > 1) {
      if( (currNy < minGridSize) || ((currNy%2) == 0) ) {
        break;
      }
      currNy = 1 + ((currNy - 1)/2); 
    }
    if(dim > 2) {
      if( (currNz < minGridSize) || ((currNz%2) == 0) ) {
        break;
      }
      currNz = 1 + ((currNz - 1)/2); 
    }
  }//lev

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny.empty());
  } else { 
    assert( (Ny.size()) == (Nx.size()) );
  }
  if(dim < 3) {
    assert(Nz.empty());
  } else {
    assert( (Nz.size()) == (Nx.size()) );
  }
#endif

  if(dim < 2) {
    Ny.resize((Nx.size()), 1);
  }
  if(dim < 3) {
    Nz.resize((Nx.size()), 1);
  }

  if(print) {
    std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
  }
}

void destroyDA(std::vector<DA>& da) {
  for(int i = 0; i < da.size(); ++i) {
    if(da[i] != NULL) {
      DADestroy(da[i]);
    }
  }//end i
  da.clear();
}

void destroyComms(std::vector<MPI_Comm> & activeComms) {
  for(int i = 0; i < activeComms.size(); ++i) {
    if(activeComms[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(activeComms[i]));
    }
  }//end i
  activeComms.clear();
}


