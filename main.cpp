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

double euclideanDistance(ArffInstance* instance1, ArffInstance* instance2, int numAttributes) {
    double sum = 0;
    for (int attributeIndex = 0; attributeIndex < (numAttributes - 1); attributeIndex++) {
        sum += pow((instance2->get(attributeIndex)->operator int32()) - (instance1->get(attributeIndex)->operator int32()), 2);
    }
    return sqrt(sum);
}

typedef struct
{
    ArffInstance*   neighbor;
    double          distance;
} NeighborDistance;

int vote(NeighborDistance* nearestNeighbors, int k, int numAttributes) {
    int* classVotes = (int *)malloc(numAttributes * sizeof(int));
    for (int i = 0; i < k; i++)
    {
        int classVote = nearestNeighbors[i].neighbor->get(numAttributes - 1)->operator int32();
        classVotes[classVote]++;
    }
    int indexOfMax = 0;
    for (int i = 0; i < numAttributes; i++)
    {
        if (classVotes[indexOfMax] < classVotes[i])
        {
            indexOfMax = i;
        }
    }
    // printf("predictedClass: %i\n",indexOfMax);
    free(classVotes);
    return indexOfMax;
}

int* KNN(ArffData* dataset, int k)
{
    int* predictions = (int *) malloc(dataset->num_instances() * sizeof(int));
    int numInstances =  dataset->num_instances();
    int numAttributes =  dataset->num_attributes();
    for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++)
    {
        ArffInstance* instance1 = dataset->get_instance(instanceIndex);
        NeighborDistance* nearestNeighbors = (NeighborDistance*) malloc(k * sizeof(NeighborDistance));

        for (int i = 0; i < k; i++)
        {
            nearestNeighbors[i].distance = FLT_MAX;
        }

        double worstBestDistance = FLT_MAX;
        for (int instance2Index = 0; instance2Index < numInstances; instance2Index++)
        {
            if (instanceIndex == instance2Index)
                continue;

            double newNeighborDistance = euclideanDistance(instance1, dataset->get_instance(instance2Index), numAttributes);
            if (instanceIndex == 0)
            {
                printf("distance was %f\n", newNeighborDistance);
            }


            bool placed = false;
            for (int i = 0; i < k; i++)
            {
                if (nearestNeighbors[i].neighbor == NULL)
                {
                    nearestNeighbors[i].distance = newNeighborDistance;
                    nearestNeighbors[i].neighbor = dataset->get_instance(instance2Index);
                    placed = true;
                    break;
                }
            }
            if (!placed) // after initial filling of nearestNeighbors... hopefully..
            {
                int indexOfMax = 0;
                double worstDistance = 0;

                for (int i = 0; i < k; i++)
                {
                    if (nearestNeighbors[i].distance > worstDistance)
                    {
                        worstDistance = nearestNeighbors[i].distance;
                        indexOfMax = i;
                    }
                }
                if (newNeighborDistance < worstDistance)
                {
                    nearestNeighbors[indexOfMax].distance = newNeighborDistance;
                    nearestNeighbors[indexOfMax].neighbor = dataset->get_instance(instance2Index);
                }
            }

        }
        predictions[instanceIndex] = vote(nearestNeighbors, k, numAttributes);
        free(nearestNeighbors);
    }

    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

    for (int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for (int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i * dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if (argc != 2)
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

