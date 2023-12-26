%We define the samples or our input K
K = 8671; 

%We then should define the doHyperparameterSearch and make it true so as to
%run the hyperparameter grid search code
doHyperparametersSearch = true;

%Defining the basic parameter we will be trying with our code
trainingRatio = [0.8 0.4 0.1];
validationRatio = [0.1 0.2 0.1];
testRatio = [0.1 0.4 0.8];
hiddenUnits = [50 200 500];

%Hyperparamater Grid Search Algorithm
if doHyperparametersSearch
    fprintf("Starting the Hyperparameters search \n")
    hiddenUnitsIdx = 1; % We will set this fixed number of hiddenUnits which is the minimum
    datasetRatiosIdx = 1; % We will set a fixed number of training/validation/test ratios
    numEpochsList = [500 1000 2000];
    learningRatesList = [0.1 0.01 0.001];
    mcRatesList = [0.1 0.2 0.3];
    
    bestNumEpochs = 0;
    bestLr = 0;
    bestMc = 0;
    bestAcc = 0;
    for i = 1:length(numEpochsList)
        for j = 1:length(learningRatesList)
            for k = 1:length(mcRatesList)
                fprintf("Begin testing using NumEpochs=%d LR=%f MC=%f\n", numEpochsList(i), learningRatesList(j), mcRatesList(k))
                [trainAcc, valAcc, testAcc] = model2(hiddenUnits(hiddenUnitsIdx), trainingRatio(datasetRatiosIdx), validationRatio(datasetRatiosIdx), testRatio(datasetRatiosIdx), X , Y, numEpochsList(i), learningRatesList(j), mcRatesList(k));
                fprintf("We will obtain the accuracy = %f\n", valAcc)
                if valAcc > bestAcc
                   bestAcc = valAcc;
                   bestNumEpochs = numEpochsList(i);
                   bestLr = learningRatesList(j);
                   bestMc = mcRatesList(k);
                end
            end
        end
    end
    fprintf("Best Hyperparamters: NumEpochs=%d LR=%f MC=%f. With %f accuracy\n", bestNumEpochs, bestLr, bestMc, bestAcc)
    numEpochs = bestNumEpochs;
    learningRate = bestLr;
    momentum = bestMc;
end



%Applying the model2 algorithm
fprintf("Begin testing with Model 2 \n")
for i = 1:length(hiddenUnits)
    for j = 1:length(trainingRatio)
        fprintf("Testing using Hidden units=%d TrainDataRatio=%f ValDataRatio=%f TestDataRatio=%f \n", hiddenUnits(i), trainingRatio(j), validationRatio(j), testRatio(j))
        [trainAcc, valAcc, testAcc] = model2(hiddenUnits(i), trainingRatio(j), validationRatio(j), testRatio(j), X, Y, numEpochs, learningRate, momentum);
        fprintf(" Accuracies: Train=%f Validation=%f Test=%f\n", trainAcc, valAcc, testAcc)
    end
end

