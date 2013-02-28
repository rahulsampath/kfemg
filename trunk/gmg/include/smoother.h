
#ifndef __SMOOTHER__
#define __SMOOTHER__

struct SmootherData {
};

void setupSmootherData(SmootherData* data);

void destroySmootherData(SmootherData* data);

void applySmoother(SmootherData* data, Vec in, Vec out);

#endif

