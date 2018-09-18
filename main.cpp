#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <limits.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

double euclideanDistance(ArffInstance* instance1, ArffInstance* instance2, int numAttributes){
    double sum=0;
    for (int attributeIndex=0; attributeIndex < numAttributes; attributeIndex++){
        sum+=pow((instance1->get(attributeIndex)->operator int32())-(instance2->get(attributeIndex)->operator int32()), 2);
    }
    return sqrt(sum);
}

int* NN(ArffData* dataset)
{
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    printf("The number of instances is %d\n", dataset->num_instances());
    printf("The number of attributes is %d\n", dataset->num_attributes());
    for (int instanceIndex=0; instanceIndex < dataset->num_instances(); instanceIndex++){
        double bestDistance=FLT_MAX;
        ArffInstance* nearestNeighbor;
        for (int instance2Index=0; instance2Index < dataset->num_instances(); instance2Index++){
            if (instanceIndex == instance2Index){
                continue;
            }
            double distance = euclideanDistance(dataset->get_instance(instanceIndex),dataset->get_instance(instance2Index),dataset->num_attributes()-1);
            // We've found a closer neighbor so lets record
            if(distance < bestDistance){
                bestDistance = distance;
                nearestNeighbor = dataset->get_instance(instance2Index);
            }
        }
        int prediction = nearestNeighbor->get(dataset->num_attributes()-1)->operator int32();
        // Put class prediction into array
        predictions[instanceIndex] = prediction;
        int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    }
    
    return predictions;
}
typedef struct
{
    ArffInstance*   neighbor;
    double          distance;
} NeighborDistance;

int isCloser(NeighborDistance* nearestNeighbors, double newNeighborDistance, int numNearestNeighbors)
{
    for (int i = 0; i < numNearestNeighbors; ++i)
    {
        if (nearestNeighbors[i].neighbor == NULL)
        {
            return i;
        }
        else if (newNeighborDistance < nearestNeighbors[i].distance)
        {
            return i;
        }
    }
    return -1;
}

int* KNN(ArffData* dataset, int k)
{
    int* predictions = (int *) malloc(dataset->num_instances() * sizeof(int));
    int numInstances =  dataset->num_instances();
    int numAttributes =  dataset->num_attributes();
    NeighborDistance* nearestNeighbors = (NeighborDistance*) malloc(k * sizeof(NeighborDistance));
     for (int instanceIndex=0; instanceIndex < numInstances; instanceIndex++)
     {
        ArffInstance* instance1 = dataset->get_instance(instanceIndex);
        for (int instance2Index=0; instance2Index < numInstances; instance2Index++)
        {
            if (instanceIndex == instance2Index)
                continue;
            double newNeighborDistance = euclideanDistance(instance1, dataset->get_instance(instance2Index), numAttributes);
            int closer = isCloser(nearestNeighbors, newNeighborDistance, k);
            if (closer != -1)
            {
                nearestNeighbors[closer].distance = newNeighborDistance;
                nearestNeighbors[closer].neighbor = dataset->get_instance(instance2Index);
            }
            predictions[instanceIndex] = nearestNeighbors[0].neighbor->get(numAttributes-1)->operator int32();
        }
     }

    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    int* predictions = KNN(dataset, 3);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}

