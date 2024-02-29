### Setup

import dataprocess as df
import sklearn.metrics as skmetric

from keras.models import load_model

seed = 22

x_test, y_test = df.x_test, df.y_test
x_test_lstm = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

### Load model

model_dnn = load_model('./models/DNN.keras')
model_cnn = load_model('./models/CNN.keras')
model_lstm = load_model('./models/LSTM.keras')



### Model Evaluation

## DNN

dnn_evl = model_dnn.evaluate(x_test, y_test, verbose = 1)
# print('Test: %.3f' % dnn_evl[1])

dnn_pred = model_dnn.predict(x_test, verbose=1)
dnn_pred = dnn_pred[:, 0].round()

dnn_rp = skmetric.classification_report(y_test, dnn_pred)

# print(dnn_rp)

dnn_cm = skmetric.confusion_matrix(y_test, dnn_pred)
# print(dnn_cm)

# plot_confusion_matrix(cm = dnn_cm, title = 'DNN')

dnn_fpr, dnn_tpr, dnn_thresholds = skmetric.roc_curve(y_test, dnn_pred)
dnn_auc = skmetric.auc(dnn_fpr, dnn_tpr)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(dnn_fpr, dnn_tpr, label='DNN (AUC = {:.3f})'.format(dnn_auc))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve - DNN')
# plt.legend(loc='lower right')
# plt.show()

## CNN

cnn_evl = model_cnn.evaluate(x_test, y_test, verbose = 1)
# print('Test: %.3f' % cnn_evl[1])

cnn_pred = model_cnn.predict(x_test, verbose=1)
cnn_pred = cnn_pred[:, 0].round()

cnn_rp = skmetric.classification_report(y_test, cnn_pred)

# print(cnn_rp)

cnn_cm = skmetric.confusion_matrix(y_test, cnn_pred)
# print(cnn_cm)

# plot_confusion_matrix(cm = cnn_cm, title = 'CNN')

cnn_fpr, cnn_tpr, cnn_thresholds = skmetric.roc_curve(y_test, cnn_pred)
cnn_auc = skmetric.auc(cnn_fpr, cnn_tpr)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(cnn_fpr, cnn_tpr, label='CNN (AUC = {:.3f})'.format(cnn_auc))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve - CNN')
# plt.legend(loc='lower right')
# plt.show()

## LSTM

lstm_evl = model_lstm.evaluate(x_test_lstm, y_test, verbose = 1)
# print('Test: %.3f' % lstm_evl[1])

lstm_pred = model_lstm.predict(x_test_lstm, verbose=1)
lstm_pred = lstm_pred[:, 0].round()

lstm_rp = skmetric.classification_report(y_test, lstm_pred)

# print(lstm_rp)

lstm_cm = skmetric.confusion_matrix(y_test, lstm_pred)
# print(lstm_cm)

# plot_confusion_matrix(cm = lstm_cm, title = 'LSTM')

lstm_fpr, lstm_tpr, lstm_thresholds = skmetric.roc_curve(y_test, lstm_pred)
lstm_auc = skmetric.auc(lstm_fpr, lstm_tpr)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(lstm_fpr, lstm_tpr, label='LSTM (AUC = {:.3f})'.format(lstm_auc))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve - LSTM')
# plt.legend(loc='lower right')
# plt.show()



### Compare different models
    
dnn_accuracy = skmetric.accuracy_score(y_test, dnn_pred)
dnn_precision = skmetric.precision_score(y_test, dnn_pred)
dnn_f1 = skmetric.f1_score(y_test, dnn_pred)

cnn_accuracy = skmetric.accuracy_score(y_test, cnn_pred)
cnn_precision = skmetric.precision_score(y_test, cnn_pred)
cnn_f1 = skmetric.f1_score(y_test, cnn_pred)

lstm_accuracy = skmetric.accuracy_score(y_test, lstm_pred)
lstm_precision = skmetric.precision_score(y_test, lstm_pred)
lstm_f1 = skmetric.f1_score(y_test, lstm_pred)

print('DNN')
print('Accuracy: %f' % dnn_accuracy)
print('Precision: %f' % dnn_precision)
print('F1-Score: %f' % dnn_f1)
print('Confusion Matrix')
print(dnn_cm)

print('')
print('------------------------------------------')
print('')

print('CNN')
print('Accuracy: %f' % cnn_accuracy)
print('Precision: %f' % cnn_precision)
print('F1-Score: %f' % cnn_f1)
print('Confusion Matrix')
print(cnn_cm)

print('')
print('------------------------------------------')
print('')

print('LSTM')
print('Accuracy: %f' % lstm_accuracy)
print('Precision: %f' % lstm_precision)
print('F1-Score: %f' % lstm_f1)
print('Confusion Matrix')
print(lstm_cm)

print('')
    






