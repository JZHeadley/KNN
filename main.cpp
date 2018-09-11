#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

int* KNN(ArffData* dataset)
{
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    printf("The number of instances is %d\n", dataset->num_instances());
    printf("The number of attributes is %d\n", dataset->num_attributes());
    for (int instanceIndex=0; instanceIndex < dataset->num_instances(); instanceIndex++){
        for(int attributeIndex=0;attributeIndex < dataset->num_attributes(); attributeIndex++){
               float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
               printf("The attribute was %d\n",attributeValue);
        }
    }
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // printf("The attribute value was %d",attributeValue);
// int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement KNN here, fill array of class predictions
    
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
    
    int* predictions = KNN(dataset);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
