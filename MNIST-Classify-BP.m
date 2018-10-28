clear;
clc
images = loadMNISTImages("train-images-idx3-ubyte");
labels = loadMNISTLabels("train-labels-idx1-ubyte");
n = 60000;
eta1 = 0.01;
eta2 = 0.01;
N = 24;
input_num = 784;
output_num = 10;
b_2 = zeros(N,output_num);
b_2(1,:) = rand(1) - 0.5;
w_1 = rand(N,input_num) - 0.5;
w_11 = w_1;
b_1 = rand(N,1) - 0.5;
w_2 = rand(N,output_num) - 0.5;
w_22 = w_2;
assistS2 = zeros(output_num , output_num);
assistS1 = zeros(N , N);
final_output = zeros(output_num,n);
%e = zeros(1,n);
count = 1;
minError = 1e-5;
sumMSE1 = 1;
sumMSE2 = 1;
%change = zeros(100,1);
j = 1;
errors=zeros(1,n);
errors_test = zeros(1,n);


    
while(1)
  
  sumMSE1 = 0;
  %%%%%%%test_train
  for i = 1:n
     
    middle_input =  w_1 * images(:,i) + b_1;
    middle_output = tanh(middle_input);
    final_input = w_2' * middle_output;
    final_output1(:,i) = tanh(final_input);
     [M,I] = max(final_output1(:,i));
    lab(i) = I-1;
    
    %a1=a1+1
   
     if lab(i) ~= labels(i,:)
       % a2=a2+1
        errors(1,count) = errors(1,count) + 1;
    end
  end
   
 error_rate(count) = errors(1,count)/n;
 %%%%%%test_test
 images_test = loadMNISTImages("t10k-images-idx3-ubyte");
labels_test = loadMNISTLabels("t10k-labels-idx1-ubyte");
m = 10000;
sumMSE1_test = 0;
 for i = 1:m
     d = zeros(10,1);
    e = zeros(10,1);
    middle_input =  w_1 * images_test(:,i) + b_1;
    middle_output = tanh(middle_input);
    final_input = w_2' * middle_output;
    final_output1(:,i) = tanh(final_input);
     [M,I] = max(final_output1(:,i));
    lab(i) = I-1;
      d(labels(i,1)+1,1) = 1;
    e = d - final_output1(:,i);
     sumMSE1_test = sumMSE1_test+ e' * e;
    %a1=a1+1
   
     if lab(i) ~= labels_test(i,:)
       % a2=a2+1
        errors_test(1,count) = errors_test(1,count) + 1;
    end
 end
  sumMSE_test(count) = sumMSE1_test;
   error_rate_test(count) = errors_test(1,count) /m;
 %%%decide if jump out
    if error_rate(count) <= 0.05
    
    break;

    end
  
for i = 1:n
    d = zeros(10,1);
    e = zeros(10,1);
    middle_input =  w_1 * images(:,i) + b_1;
    middle_output = tanh(middle_input);
    
    final_input = w_2' * middle_output;
    final_output(:,i) =tanh(final_input);
   
    d(labels(i,1)+1,1) = 1;
    
    assistS1 = diag(1 - (middle_output .* middle_output));
    assistS2 = diag(1 - (final_output(:,i) .* final_output(:,i)));
    e = d - final_output(:,i);
    s2 = -2 * assistS2 * e; 
    s1 = assistS1 * w_2 * s2; 
    w_2 = w_2 - eta2 * middle_output * s2';
    b_2(1,:) = b_2(1,:) - eta2 * s2'; 
    w_1 = w_1 - eta1 * s1 * images(:,i)';
    b_1 = b_1 - eta1 * s1;
    sumMSE1 = sumMSE1+ e' * e;
   
   
end
    sumMSE(count) = sumMSE1;
    if (count>2 && sumMSE2<sumMSE(count));
        eta = eta*0.9;

    change(j) = count;
    j = j + 1;
    end
    sumMSE2 = sumMSE(count);
  if sumMSE(count) <= minError
      break;
  end
    count=count+1;
end
figure(1)
plot(1:count,error_rate);
title "trian--errorRate";
figure(2)
plot(1:count, error_rate_test);
title "test--errorRate";
figure(3)
plot(1:count-1,sumMSE(1:count-1));
title "train--energy";
figure(4)
plot(1:count,sumMSE_test(1:count));
title "test--energy";

