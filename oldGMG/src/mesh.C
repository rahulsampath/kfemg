
#include "gmg/include/mesh.h"
#include <iostream>
#include <cmath>

#ifdef DEBUG
#include <cassert>
#endif

void createDA(int dim, int dofsPerNode, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY, 
    std::vector<std::vector<PetscInt> >& partX, std::vector<int>& activeNpes, std::vector<MPI_Comm>& activeComms, 
    std::vector<DM>& da) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int nlevels = Nx.size();
  da.clear();
  da.resize(nlevels, NULL);
  if(dim == 1) {
    for(int lev = 0; lev < nlevels; ++lev) {
      if(rank < (activeNpes[lev])) {
        DMDACreate1d(activeComms[lev], DMDA_BOUNDARY_NONE, Nx[lev], dofsPerNode, 1, &(partX[lev][0]), &(da[lev]));
      }
    }//end lev
  } else if(dim == 2) {
    for(int lev = 0; lev < nlevels; ++lev) {
      if(rank < (activeNpes[lev])) {
        DMDACreate2d(activeComms[lev], DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX,
            Nx[lev], Ny[lev], (partX[lev].size()), (partY[lev].size()), dofsPerNode, 1,
            &(partX[lev][0]), &(partY[lev][0]), &(da[lev]));
      }
    }//end lev
  } else {
    for(int lev = 0; lev < nlevels; ++lev) {
      if(rank < (activeNpes[lev])) {
        DMDACreate3d(activeComms[lev], DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
            DMDA_STENCIL_BOX, Nx[lev], Ny[lev], Nz[lev], (partX[lev].size()), (partY[lev].size()),
            (partZ[lev].size()), dofsPerNode, 1, &(partX[lev][0]), &(partY[lev][0]),
            &(partZ[lev][0]), &(da[lev]));
      }
    }//end lev
  }
}

void createActiveComms(std::vector<int>& activeNpes, std::vector<MPI_Comm>& activeComms) {
  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* rankList = new int[npes];
  for(int i = 0; i < npes; ++i) {
    rankList[i] = i;
  }//end for i

  MPI_Group group;
  MPI_Comm_group(MPI_COMM_WORLD, &group);

  int nlevels = activeNpes.size();

  activeComms.resize(nlevels);
  for(int lev = 0; lev < nlevels; ++lev) {
    if(rank < (activeNpes[lev])) {
      MPI_Group subGroup;
      MPI_Group_incl(group, (activeNpes[lev]), rankList, &subGroup);
      MPI_Comm_create(MPI_COMM_WORLD, subGroup, &(activeComms[lev]));
      MPI_Group_free(&subGroup);
    } else {
      MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_EMPTY, &(activeComms[lev]));
    }
  }//end lev

  MPI_Group_free(&group);

  delete [] rankList;
}

void computePartition(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ, 
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, 
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<PetscInt> >& scanZ,
    std::vector<std::vector<PetscInt> >& scanY, std::vector<std::vector<PetscInt> >& scanX,
    std::vector<int>& activeNpes, bool print) {
  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int nlevels = Nx.size();

  PetscInt maxCoarseNpes = npes;
  PetscOptionsGetInt(PETSC_NULL, "-maxCoarseNpes", &maxCoarseNpes, PETSC_NULL);
  if(dim == 1) {
    partX.resize(nlevels);
    scanX.resize(nlevels);
    offsets.resize(nlevels);
    int maxNpes = npes;
    for(int lev = (nlevels - 1); lev > 0; --lev) {
      computePartition1D(Nx[lev], maxNpes, partX[lev], offsets[lev], scanX[lev]);
      maxNpes = offsets[lev].size();
    }//end lev
    if(maxCoarseNpes > maxNpes) {
      maxCoarseNpes = maxNpes;
    }
    computePartition1D(Nx[0], maxCoarseNpes, partX[0], offsets[0], scanX[0]);
    if(print) {
      for(int lev = 0; lev < nlevels; ++lev) {
        std::cout<<"Lev = "<<lev<<", px = "<<(partX[lev].size())<<std::endl;
      }//end lev
    }
  } else if(dim == 2) {
    partX.resize(nlevels);
    partY.resize(nlevels);
    scanX.resize(nlevels);
    scanY.resize(nlevels);
    offsets.resize(nlevels);
    int maxNpes = npes;
    for(int lev = (nlevels - 1); lev > 0; --lev) {
      computePartition2D(Ny[lev], Nx[lev], maxNpes, partY[lev], partX[lev], offsets[lev], scanY[lev], scanX[lev]);
      maxNpes = offsets[lev].size();
    }//end lev
    if(maxCoarseNpes > maxNpes) {
      maxCoarseNpes = maxNpes;
    }
    computePartition2D(Ny[0], Nx[0], maxCoarseNpes, partY[0], partX[0], offsets[0], scanY[0], scanX[0]);
    if(print) {
      for(int lev = 0; lev < nlevels; ++lev) {
        std::cout<<"Lev = "<<lev<<", px = "<<(partX[lev].size())<<", py = "<<(partY[lev].size())<<std::endl;
      }//end lev
    }
  } else {
    partX.resize(nlevels);
    partY.resize(nlevels);
    partZ.resize(nlevels);
    scanX.resize(nlevels);
    scanY.resize(nlevels);
    scanZ.resize(nlevels);
    offsets.resize(nlevels);
    int maxNpes = npes;
    for(int lev = (nlevels - 1); lev > 0; --lev) {
      computePartition3D(Nz[lev], Ny[lev], Nx[lev], maxNpes, partZ[lev], partY[lev], partX[lev], 
          offsets[lev], scanZ[lev], scanY[lev], scanX[lev]);
      maxNpes = offsets[lev].size();
    }//end lev
    if(maxCoarseNpes > maxNpes) {
      maxCoarseNpes = maxNpes;
    }
    computePartition3D(Nz[0], Ny[0], Nx[0], maxCoarseNpes, partZ[0], partY[0], partX[0], 
        offsets[0], scanZ[0], scanY[0], scanX[0]);
    if(print) {
      for(int lev = 0; lev < nlevels; ++lev) {
        std::cout<<"Lev = "<<lev<<", px = "<<(partX[lev].size())
          <<", py = "<<(partY[lev].size())<<", pz = "<<(partZ[lev].size())<<std::endl;
      }//end lev
    }
  }

  activeNpes.resize(nlevels);
  for(int lev = 0; lev < nlevels; ++lev) {
    activeNpes[lev] = offsets[lev].size();
  }//end lev
}

void createGrids(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, bool print) {
  if(dim == 1) {
    createGrids1D(Nx, print);
  } else if(dim == 2) {
    createGrids2D(Ny, Nx, print);
  } else {
    createGrids3D(Nz, Ny, Nx, print);
  }
}

void computePartition3D(PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt>& partZ, std::vector<PetscInt>& partY, std::vector<PetscInt>& partX,
    std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanZ, 
    std::vector<PetscInt>& scanY, std::vector<PetscInt>& scanX) {
  PetscInt px = static_cast<PetscInt>(std::pow(maxNpes, (1.0/3.0)));
  if(px > Nx) {
    px = Nx;
  }

  PetscInt py = static_cast<PetscInt>(std::sqrt(maxNpes/px));
  if(py > Ny) {
    py = Ny;
  }

  PetscInt pz = maxNpes/(px*py);
  if(pz > Nz) {
    pz = Nz;
  }

  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  partX.clear();  
  partX.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(partX[cnt]);
  }//end cnt

  PetscInt avgY = Ny/py;
  PetscInt extraY = Ny%py; 
  partY.clear();  
  partY.resize(py, avgY);
  for(int cnt = 0; cnt < extraY; ++cnt) {
    ++(partY[cnt]);
  }//end cnt

  PetscInt avgZ = Nz/pz;
  PetscInt extraZ = Nz%pz; 
  partZ.clear();  
  partZ.resize(pz, avgZ);
  for(int cnt = 0; cnt < extraZ; ++cnt) {
    ++(partZ[cnt]);
  }//end cnt

  scanX.clear();  
  scanX.resize(px);
  scanX[0] = partX[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanX[i] = scanX[i - 1] + partX[i];
  }//end i

  scanY.clear();  
  scanY.resize(py);
  scanY[0] = partY[0] - 1;
  for(int i = 1; i < py; ++i) {
    scanY[i] = scanY[i - 1] + partY[i];
  }//end i

  scanZ.clear();  
  scanZ.resize(pz);
  scanZ[0] = partZ[0] - 1;
  for(int i = 1; i < pz; ++i) {
    scanZ[i] = scanZ[i - 1] + partZ[i];
  }//end i

  offsets.clear();
  offsets.resize(px*py*pz);
  offsets[0] = 0;
  for(int p = 1; p < (px*py*pz); ++p) {
    int i = (p - 1)%px;
    int j = ((p - 1)/px)%py;
    int k = (p - 1)/(px*py);
    offsets[p] = offsets[p - 1] + (partX[i] * partY[j] * partZ[k]);
  }//end p
}

void computePartition2D(PetscInt Ny, PetscInt Nx, int maxNpes, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanY, 
    std::vector<PetscInt>& scanX) {
  PetscInt px = static_cast<PetscInt>(std::sqrt(maxNpes));
  if(px > Nx) {
    px = Nx;
  }

  PetscInt py = maxNpes/px;
  if(py > Ny) {
    py = Ny;
  }

  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  partX.clear();  
  partX.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(partX[cnt]);
  }//end cnt

  PetscInt avgY = Ny/py;
  PetscInt extraY = Ny%py; 
  partY.clear();  
  partY.resize(py, avgY);
  for(int cnt = 0; cnt < extraY; ++cnt) {
    ++(partY[cnt]);
  }//end cnt

  scanX.clear();  
  scanX.resize(px);
  scanX[0] = partX[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanX[i] = scanX[i - 1] + partX[i];
  }//end i

  scanY.clear();  
  scanY.resize(py);
  scanY[0] = partY[0] - 1;
  for(int i = 1; i < py; ++i) {
    scanY[i] = scanY[i - 1] + partY[i];
  }//end i

  offsets.clear();
  offsets.resize(px*py);
  offsets[0] = 0;
  for(int p = 1; p < (px*py); ++p) {
    int i = (p - 1)%px;
    int j = (p - 1)/px;
    offsets[p] = offsets[p - 1] + (partX[i] * partY[j]);
  }//end p
}

void computePartition1D(PetscInt Nx, int maxNpes, std::vector<PetscInt>& partX, 
    std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanX) {
  PetscInt px = maxNpes;
  if(px > Nx) {
    px = Nx;
  }

  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  partX.clear();  
  partX.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(partX[cnt]);
  }//end cnt

  scanX.clear();  
  scanX.resize(px);
  scanX[0] = partX[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanX[i] = scanX[i - 1] + partX[i];
  }//end i

  offsets.clear();
  offsets.resize(px);
  offsets[0] = 0;
  for(int i = 1; i < px; ++i) {
    offsets[i] = offsets[i - 1] + partX[i - 1];
  }//end i
}

void createGrids1D(std::vector<PetscInt>& Nx, bool print) {
  PetscInt currNx = 9;
  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  if(print) {
    std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
    std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;
  }
  int minGridSize = 5;
  Nx.clear();
  //0 is the coarsest level.
  for(int lev = 0; lev < maxNumLevels; ++lev) {
    Nx.insert(Nx.begin(), currNx);
    if( (currNx <= minGridSize) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
  }//lev
  if(print) {
    std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
  }
}

void createGrids2D(std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, bool print) {
  PetscInt currNx = 9;
  PetscInt currNy = 9;
  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-finestNy", &currNy, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  if(print) {
    std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
    std::cout<<"Ny (Finest) = "<<currNy<<std::endl;
    std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;
  }
  int minGridSize = 5;
  Nx.clear();
  Ny.clear();
  //0 is the coarsest level.
  for(int lev = 0; lev < maxNumLevels; ++lev) {
    Nx.insert(Nx.begin(), currNx);
    Ny.insert(Ny.begin(), currNy);
    if( (currNx <= minGridSize) || ((currNx%2) == 0) ) {
      break;
    }
    if( (currNy <= minGridSize) || ((currNy%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    currNy = 1 + ((currNy - 1)/2); 
  }//lev
  if(print) {
    std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
  }
}

void createGrids3D(std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, bool print) {
  PetscInt currNx = 9;
  PetscInt currNy = 9;
  PetscInt currNz = 9;
  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-finestNy", &currNy, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-finestNz", &currNz, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  if(print) {
    std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
    std::cout<<"Ny (Finest) = "<<currNy<<std::endl;
    std::cout<<"Nz (Finest) = "<<currNz<<std::endl;
    std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;
  }
  int minGridSize = 5;
  Nx.clear();
  Ny.clear();
  Nz.clear();
  //0 is the coarsest level.
  for(int lev = 0; lev < maxNumLevels; ++lev) {
    Nx.insert(Nx.begin(), currNx);
    Ny.insert(Ny.begin(), currNy);
    Nz.insert(Nz.begin(), currNz);
    if( (currNx <= minGridSize) || ((currNx%2) == 0) ) {
      break;
    }
    if( (currNy <= minGridSize) || ((currNy%2) == 0) ) {
      break;
    }
    if( (currNz <= minGridSize) || ((currNz%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    currNy = 1 + ((currNy - 1)/2); 
    currNz = 1 + ((currNz - 1)/2); 
  }//lev
  if(print) {
    std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
  }
}

void destroyDA(std::vector<DM>& da) {
  for(size_t i = 0; i < da.size(); ++i) {
    if(da[i] != NULL) {
      DMDestroy(&(da[i]));
    }
  }//end i
  da.clear();
}

void destroyComms(std::vector<MPI_Comm> & activeComms) {
  for(size_t i = 0; i < activeComms.size(); ++i) {
    if(activeComms[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(activeComms[i]));
    }
  }//end i
  activeComms.clear();
}


