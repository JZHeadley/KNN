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
#define DEBUG false
// #define INSTANCETOCHECK 4897
#define K 4

#define MIN(a,b) (((a)<(b))?(a):(b))
#define NUM_THREADS 8


int INSTANCETOCHECK = 1;
ArffData* dataset;
int numInstances;
int numAttributes;
int* predictions;

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

typedef struct
{
    int           threadId;
    int           k;
} KNNArgs;

int vote(NeighborDistance* nearestNeighbors, int k, int numAttributes, int instanceIndex) {
    int* classVotes = (int *)malloc(numAttributes * sizeof(int));
    for (int i = 0; i < numAttributes; i++)
    {
        classVotes[i] = 0;
    }
    for (int i = 0; i < k; i++)
    {
        int classVote = nearestNeighbors[i].neighbor->get(numAttributes - 1)->operator int32();
        if (instanceIndex == INSTANCETOCHECK && DEBUG)
        {
            printf("The class of this neigbor is %i\n", classVote);
        }
        classVotes[classVote]++;
    }
    int indexOfMax = 0;
    int countMax = 0;
    for (int i = 0; i < numAttributes; i++)
    {
        if (classVotes[i] == countMax && classVotes[i] > 0)
        {
            if (instanceIndex == INSTANCETOCHECK && DEBUG) {
                printf("seems like we've found a duplicate?\n");
            }
            // handle duplicates here or pass off to method to do so

            //lets remove the worst neighbor and revote with a lower k
            int worstNeighborIndex = 0;
            double worstDistance = 0;
            for (int j = 0; j < k; j++)
            {
                if (nearestNeighbors[j].distance > worstDistance)
                {
                    worstDistance = nearestNeighbors[j].distance;
                    worstNeighborIndex = j;
                }
            }
            if (instanceIndex == INSTANCETOCHECK && DEBUG) {
                printf("max was %f at index %i\n", worstDistance, worstNeighborIndex);
            }
            // we know the worst now so lets reconstruct the neighbors without it
            NeighborDistance* newNeighbors = (NeighborDistance*) malloc((k - 1) * sizeof(NeighborDistance));
            // I hate linked lists but heres where I regret not using one...
            int neighborIndex = 0;
            for (int j = 0; j < k; j++)
            {
                if (j != worstNeighborIndex)
                {
                    newNeighbors[neighborIndex].neighbor = nearestNeighbors[j].neighbor;
                    newNeighbors[neighborIndex].distance = nearestNeighbors[j].distance;
                    neighborIndex++;
                }
            }
            int classResult = vote(newNeighbors, k - 1, numAttributes, instanceIndex);
            // free(newNeighbors);
            return classResult;

        }
        if (classVotes[indexOfMax] < classVotes[i])
        {
            indexOfMax = i;
            countMax = classVotes[i];
        }
    }
    if (instanceIndex == INSTANCETOCHECK && DEBUG) {
        printf("The predicted class was %i\n", indexOfMax);
    }
    return indexOfMax;
}

void* KNN(void* args)
{
    KNNArgs* knnArgs = (KNNArgs*) args;
    int threadId = knnArgs->threadId;
    int k = knnArgs->k;


    int instancesPerTask = (dataset->num_instances() + NUM_THREADS) / NUM_THREADS;
    int beginIndex = threadId * instancesPerTask;
    int endIndex = MIN((threadId + 1) * instancesPerTask, dataset->num_instances());

    for (int instanceIndex = beginIndex; instanceIndex < endIndex; instanceIndex++)
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
            if (instanceIndex == INSTANCETOCHECK && DEBUG)
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
                    if (instanceIndex == INSTANCETOCHECK && DEBUG)
                    {
                        printf("new nearest has been found at position %d with first attribute %f\n", instance2Index, dataset->get_instance(instance2Index)->get(0)->operator float());
                    }
                    nearestNeighbors[indexOfMax].distance = newNeighborDistance;
                    nearestNeighbors[indexOfMax].neighbor = dataset->get_instance(instance2Index);
                }
            }

        }
        int classification = vote(nearestNeighbors, k, numAttributes, instanceIndex);
        predictions[instanceIndex] = classification;
        // delete(knnArgs);
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
        if (argc != 3)
        {
            cout << "Usage: ./main datasets/datasetFile.arff" << endl;
            exit(0);
        }
    }

    ArffParser parser(argv[1]);
    dataset = parser.parse();
    struct timespec start, end;

    numInstances = dataset->num_instances();
    numAttributes = dataset->num_attributes();

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    if (argv[2] != NULL)
    {
        INSTANCETOCHECK = atoi(argv[2]);
    }

    predictions = (int *) malloc(numInstances * sizeof(int));

    pthread_t *threads = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    int* threadIds = (int*)malloc(NUM_THREADS * sizeof(int));

    for (int i = 0; i < NUM_THREADS; i++)
        threadIds[i] = i;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        KNNArgs* args = new KNNArgs;
        args->threadId = threadIds[i];
        args->k = K;
        int status = pthread_create(&threads[i], NULL, KNN,  (void*) args);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    free(predictions);
    free(threadIds);
    free(threads);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}

