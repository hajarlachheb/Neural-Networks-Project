function [trainAcc, valAcc, testAcc] = model2(hiddenUnits, trainingRatio, validationRatio, testRatio, X, Y, epochs, lr, mc)
%LOGSIGMOID + SOFTMAX + CROSS ENTHROPY
    trainAcc = 0;
    valAcc = 0;
    testAcc = 0;
    numTests = 10;
    for j = 1:numTests
        net = feedforwardnet([hiddenUnits]); % We select the number of hidden units (we have a set of them)
        for i =1:(length(net.layers)-1)
            net.layers{i}.transferFcn = 'logsig'; % We are defining LogSig that we are supposed to use in our task 
        end
        net.layers{end}.transferFcn = 'softmax'; % We are defining SoftMax that we are supposed to use in our task 
        end
        net.performFcn = 'crossentropy'; %Defining the crossenthropy
        net.trainFcn = 'traingdx'; % We will work with the train function: Gradient descent with momentum
        net.trainParam.lr = lr; 
        net.trainParam.mc = mc; 
        net.trainParam.epochs = epochs;
        net.outputs{end}.processFcns = {}; 
        net.divideFcn = 'dividerand'; % This is used to divide randomly the dataset
        net.divideParam.trainRatio = trainingRatio; % Ratio of data we wil be using as training set
        net.divideParam.valRatio = validationRatio; % Ratio of data we wil be using  as validation set
        net.divideParam.testRatio = testRatio; % Ratio of data we wil be using  as test set
        net.trainParam.max_fail = 6; % We tried many values and choose 6, because it performs better 
        net.trainParam.min_grad = 1e-5;
        [tr,F] = train(net,X,Y);
        [argmaxT] = max(Y);
        [argmaxY] = max(F);
        trainAcc = trainAcc + (sum(argmaxT(tr.trainInd)==argmaxY(tr.trainInd)) / length(tr.trainInd))/numTests;
        valAcc = valAcc + (sum(argmaxT(tr.valInd)==argmaxY(tr.valInd)) / length(tr.valInd))/numTests;
        testAcc = testAcc + (sum(argmaxT(tr.testInd)==argmaxY(tr.testInd)) / length(tr.testInd))/numTests;
    end
end 