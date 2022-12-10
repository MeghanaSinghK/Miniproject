clear all;close all;clc;
srcFiles = dir('Z:\2021\BUSINESS\Fruit\Dataset\mango\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('Z:\2021\BUSINESS\Fruit\Dataset\mango\',srcFiles(i).name);
im = imread(filename);


im = imresize(im,[256 256]);
Gray = rgb2gray(im);


newfilename=fullfile('Z:\2021\BUSINESS\Fruit\Dataset\man\',['NT' num2str(i) '.jpg']);
imwrite(Gray,newfilename,'jpg');
end