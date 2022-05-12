
[x, t] = wine_dataset;
x_axis = (2:10);
y_axis = zeros([1, length(x_axis)]);

for i = 1:length(x_axis)
    xval = x_axis(i);
    y_axis(i) = calculateAcc(x,t,x_axis(i)/10);
end
disp(x_axis)
disp(y_axis)

plot(x_axis, y_axis)

function accuracy = calculateAcc(x,t,trainColumns)
    trainFcn = 'trainscg';
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    net.trainParam.epochs = 10;
    net.trainParam.showWindow=0;
    
    [m, ~] = size(x);
    columnsNeeded = int8(m*trainColumns);
    
    xMod = x(randperm(m, columnsNeeded),:);
    
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;
    
    [net,~] = train(net,xMod,t);
    
    y = net(xMod);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    accuracy = 1 - percentErrors;
end