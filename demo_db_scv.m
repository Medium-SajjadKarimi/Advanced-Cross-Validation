
clc
clear
close all

rng(42)
C1_count = 40;
C2_count = 10;

X1 = randn(C1_count,2); % two dimensional features
X2 = [2+0.5*randn(floor(C2_count/2),2);-2+0.5*randn(ceil(C2_count/2),2)];

% Generate class labels
class_labels = [ones(C1_count,1); 2*ones(C2_count,1)]; % Class 1 and Class 2 labels
X = [X1; X2]; % Combine feature matrices


figure(Position=[100,100, 1500, 200]);
sub_plt(1) = subplot(1,6,1);
scatter(X(class_labels==1, 1), X(class_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
hold on
grid on
scatter(X(class_labels==2, 1), X(class_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
title(['All samples'],'Interpreter' ,'latex','FontSize',16)
grid on
set(gca, 'FontWeight','bold','FontSize',10);
xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
ylabel("x2",'FontSize',14,'Interpreter' ,'latex')


%% Stratified K-Fold Cross-Validation

% Create a 5-fold partition
cv = cvpartition(class_labels, "KFold", 5,"Stratify",false);

perf_curve_outputs = [];

% Loop through each fold
for fold = 1:cv.NumTestSets
    % Get training and test indices for the current fold
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    test_labels = class_labels(test_idx);
    X_test = X(test_idx,:);

    sub_plt(fold+1) = subplot(1,6,fold+1);
    scatter(X_test(test_labels==1, 1), X_test(test_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
    hold on
    grid on
    scatter(X_test(test_labels==2, 1), X_test(test_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

    % legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
    grid on
    set(gca, 'FontWeight','bold','FontSize',10);
    xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
    ylabel("x2",'FontSize',14,'Interpreter' ,'latex')
    title(['Fold: ',num2str(fold)],'Interpreter' ,'latex','FontSize',14)

    % Extract training and test data
    X_train = X(train_idx, :);
    y_train = class_labels(train_idx);
    X_test = X(test_idx, :);
    y_test = class_labels(test_idx);

    % Train KNN classifier with K=1
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 1);

    % Predict class scores (posterior probabilities)
    [~, scores] = predict(knnModel, X_test);

    perf_curve_outputs = [perf_curve_outputs;[y_test, scores(:,2)]];


end

% Compute AUC for this fold
[~,~,~,auc_values] = perfcurve(perf_curve_outputs(:,1), perf_curve_outputs(:,2), 2);

linkaxes(sub_plt,'xy')
% Compute and display average AUC
avg_auc = mean(auc_values);
fprintf("AUC over 5 folds: %.4f\n", avg_auc);

saveas(gca,['kfoldCV.png'])


%% Stratified K-Fold Cross-Validation

figure(Position=[100,100, 1500, 200]);
sub_plt(1) = subplot(1,6,1);
scatter(X(class_labels==1, 1), X(class_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
hold on
grid on
scatter(X(class_labels==2, 1), X(class_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
title(['All samples'],'Interpreter' ,'latex','FontSize',16)
grid on
set(gca, 'FontWeight','bold','FontSize',10);
xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
ylabel("x2",'FontSize',14,'Interpreter' ,'latex')



% Create a 5-fold stratified partition
cv = cvpartition(class_labels, "KFold", 5,"Stratify",true);

perf_curve_outputs = [];

% Loop through each fold
for fold = 1:cv.NumTestSets
    % Get training and test indices for the current fold
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    test_labels = class_labels(test_idx);
    X_test = X(test_idx,:);

    sub_plt(fold+1) = subplot(1,6,fold+1);
    scatter(X_test(test_labels==1, 1), X_test(test_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
    hold on
    grid on
    scatter(X_test(test_labels==2, 1), X_test(test_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

    % legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
    grid on
    set(gca, 'FontWeight','bold','FontSize',10);
    xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
    ylabel("x2",'FontSize',14,'Interpreter' ,'latex')
    title(['Fold: ',num2str(fold)],'Interpreter' ,'latex','FontSize',14)

    % Extract training and test data
    X_train = X(train_idx, :);
    y_train = class_labels(train_idx);
    X_test = X(test_idx, :);
    y_test = class_labels(test_idx);

    % Train KNN classifier with K=1
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 1);

    % Predict class scores (posterior probabilities)
    [~, scores] = predict(knnModel, X_test);

    perf_curve_outputs = [perf_curve_outputs;[y_test, scores(:,2)]];


end

% Compute AUC for this fold
[~,~,~,auc_values] = perfcurve(perf_curve_outputs(:,1), perf_curve_outputs(:,2), 2);

linkaxes(sub_plt,'xy')
% Compute and display average AUC
avg_auc = mean(auc_values);
fprintf("AUC over 5 folds: %.4f\n", avg_auc);

saveas(gca,['kfoldSCV.png'])

%% Distribution-Balanced Stratified Cross-Validation

cv = dobscv(X, class_labels, 5);

figure(Position=[100,100, 1500, 200]);
sub_plt(1) = subplot(1,6,1);
scatter(X(class_labels==1, 1), X(class_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
hold on
grid on
scatter(X(class_labels==2, 1), X(class_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
title(['All samples'],'Interpreter' ,'latex','FontSize',16)
grid on
set(gca, 'FontWeight','bold','FontSize',10);
xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
ylabel("x2",'FontSize',14,'Interpreter' ,'latex')


perf_curve_outputs = [];

% Loop through each fold
for fold = 1:cv(1).NumTestSets
    % Get training and test indices for the current fold
    % train_idx = dbscv_out~=fold-1;
    % test_idx = dbscv_out==fold-1;

    train_idx = cv(fold).train_idx;
    test_idx = cv(fold).test_idx;

    test_labels = class_labels(test_idx);
    X_test = X(test_idx,:);

    sub_plt(fold+1) = subplot(1,6,fold+1);
    scatter(X_test(test_labels==1, 1), X_test(test_labels==1, 2),25, 'b*', 'LineWidth', 1.5);
    hold on
    grid on
    scatter(X_test(test_labels==2, 1), X_test(test_labels==2, 2), 25, 'ro', 'LineWidth', 1.5);

    % legend([{'Class 1'},{'Class 2'}],'Interpreter' ,'latex','FontSize',10,'Location','southeast')
    grid on
    set(gca, 'FontWeight','bold','FontSize',10);
    xlabel("x1",'FontSize',14,'Interpreter' ,'latex')
    ylabel("x2",'FontSize',14,'Interpreter' ,'latex')
    title(['Fold: ',num2str(fold)],'Interpreter' ,'latex','FontSize',14)

    % Extract training and test data
    X_train = X(train_idx, :);
    y_train = class_labels(train_idx);
    X_test = X(test_idx, :);
    y_test = class_labels(test_idx);

    % Train KNN classifier with K=1
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 1);

    % Predict class scores (posterior probabilities)
    [~, scores] = predict(knnModel, X_test);

    perf_curve_outputs = [perf_curve_outputs;[y_test, scores(:,2)]];


end

% Compute AUC for this fold
[~,~,~,auc_values] = perfcurve(perf_curve_outputs(:,1), perf_curve_outputs(:,2), 2);

linkaxes(sub_plt,'xy')
% Compute and display average AUC
avg_auc = mean(auc_values);
fprintf("AUC over 5 folds: %.4f\n", avg_auc);

saveas(gca,['DBSCV.png'])