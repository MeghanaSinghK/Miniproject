clc;
clear;
close all;
close all hidden;

[file,path] = uigetfile('*.*');
a = imread([path,file]);
figure;imshow(a);title('Input Image')
b = imresize(a,[256 256]);
b = imnoise(b,'salt & pepper');
b = medfilt3(b);
b = rgb2gray(b)

% % Dataset 

imdsTrain = imageDatastore("Z:\2021\BUSINESS\Fruit\Code\Dataset","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.7);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([256 256 1],imdsTrain);
augimdsValidation = augmentedImageDatastore([256 256 1],imdsValidation);
opts = trainingOptions("sgdm",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",45,...
    "MiniBatchSize",64,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
layers = [
    imageInputLayer([256 256 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);

YPred = classify(net,b);
output=char(YPred);
msgbox(output)

disp('Final Taining Accuracy:');
disp(traininfo.TrainingAccuracy(opts.MaxEpochs));
