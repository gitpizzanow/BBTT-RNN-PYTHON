#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define lr 0.005f
#define EPOCHS 200
#define CLIP(x) (x > 5.0 ? 5.0 : (x < -5.0 ? -5.0 : x))
#define IDX(i, j, ncols) ((i) * (ncols) + (j))  // index
#define RAND_WEIGHT() ((double)rand() / RAND_MAX * 2.0 - 1.0)

typedef struct {
  int input_size;
  int hidden_size;
  int output_size;

  double* wxh;
  double* whh;
  double* why;
  double* bxh;
  double* by;
  double* h;
  double* h_row;
  double* y_pred;

  int time_steps;

} RNN;

void freeRNN(RNN rnn);
double loss(double y_true, double y_pred);
double tanh_derivative(double x);
double loss_derivative(double y_true, double y_pred);
void forward(RNN rnn, double* X, double* Y, int n);
void backward(RNN rnn, double* X, double* Y, int n, double* dwxh, double* dwhh,
              double* dwhy, double* dbxh, double* dby, double* dh_next,
              double* dloss_out);
void apply_gradients(RNN rnn, double* dwxh, double* dwhh, double* dwhy,
                     double* dbxh, double* dby, double* dh_next);
double mean_squared_error(double* y_true, double* y_pred, int size);
double mean_absolute_error(double* y_true, double* y_pred, int size);
double regression_accuracy(double* y_true, double* y_pred, int size,
                           double tolerance);
RNN create_rnn(int input_size, int hidden_size, int output_size,
               int time_steps);

RNN create_rnn(int input_size, int hidden_size, int output_size,
               int time_steps) {
  RNN rnn;
  rnn.input_size = input_size;
  rnn.hidden_size = hidden_size;
  rnn.output_size = output_size;
  rnn.time_steps = time_steps;

  rnn.wxh = malloc(sizeof(double) * hidden_size * input_size);
  rnn.whh = malloc(sizeof(double) * hidden_size * hidden_size);
  rnn.why = malloc(sizeof(double) * output_size * hidden_size);
  rnn.bxh = malloc(sizeof(double) * hidden_size);
  rnn.by = malloc(sizeof(double) * output_size);

  rnn.h = calloc((time_steps + 1) * hidden_size, sizeof(double));
  rnn.y_pred = malloc(sizeof(double) * time_steps * output_size);

  for (int i = 0; i < hidden_size * input_size; i++) rnn.wxh[i] = RAND_WEIGHT();
  for (int i = 0; i < hidden_size * hidden_size; i++)
    rnn.whh[i] = RAND_WEIGHT();
  for (int i = 0; i < output_size * hidden_size; i++)
    rnn.why[i] = RAND_WEIGHT();
  for (int i = 0; i < hidden_size; i++) rnn.bxh[i] = RAND_WEIGHT();
  for (int i = 0; i < output_size; i++) rnn.by[i] = RAND_WEIGHT();

  return rnn;
}

void freeRNN(RNN rnn) {
  free(rnn.wxh);
  free(rnn.whh);
  free(rnn.why);
  free(rnn.bxh);
  free(rnn.by);
  free(rnn.h);
  free(rnn.h_row);
  free(rnn.y_pred);
}

double loss(double y_true, double y_pred) {
  double l = y_true - y_pred;
  return l * l;
}

double tanh_derivative(double x) {
  return 1.0f - x * x;  // x is tanh
}

double loss_derivative(double y_true, double y_pred) {
  return 2.0f * (y_pred - y_true);
}
void forward(RNN rnn, double* X, double* target, int n) {
  for (int t = 0; t < n; t++) {
    for (int i = 0; i < rnn.hidden_size; i++) {
      double sum = rnn.bxh[i];

      for (int j = 0; j < rnn.input_size; j++)
        sum +=
            rnn.wxh[IDX(i, j, rnn.input_size)] * X[IDX(t, j, rnn.input_size)];

      for (int k = 0; k < rnn.hidden_size; k++)
        sum += rnn.whh[IDX(i, k, rnn.hidden_size)] *
               rnn.h[IDX(t, k, rnn.hidden_size)];

      rnn.h[IDX(t + 1, i, rnn.hidden_size)] = tanh(sum);
    }

    double out = rnn.by[0];
    for (int i = 0; i < rnn.hidden_size; i++)
      out += rnn.why[i] * rnn.h[IDX(t + 1, i, rnn.hidden_size)];
    rnn.y_pred[t] = out;
  }
}

void backward(RNN rnn, double* X, double* target, int n, double* dwxh,
              double* dwhh, double* dwhy, double* dbxh, double* dby,
              double* dh_next, double* dloss_out) {
  for (int t = n - 1; t >= 0; t--) {
    double dy = rnn.y_pred[t] - target[t];
    dloss_out[t] = dy;

    for (int i = 0; i < rnn.hidden_size; i++) {
      dwhy[i] += dy * rnn.h[IDX(t + 1, i, rnn.hidden_size)];
    }
    dby[0] += dy;

    for (int i = 0; i < rnn.hidden_size; i++) {
      double dh = dy * rnn.why[i] + dh_next[i];
      double h_raw = rnn.h[IDX(t + 1, i, rnn.hidden_size)];
      double dh_raw = dh * tanh_derivative(h_raw);

      for (int j = 0; j < rnn.input_size; j++)
        dwxh[IDX(i, j, rnn.input_size)] +=
            dh_raw * X[IDX(t, j, rnn.input_size)];

      for (int k = 0; k < rnn.hidden_size; k++)
        dwhh[IDX(i, k, rnn.hidden_size)] +=
            dh_raw * rnn.h[IDX(t, k, rnn.hidden_size)];

      dbxh[i] += dh_raw;
      dh_next[i] = 0.0;
      for (int k = 0; k < rnn.hidden_size; k++)
        dh_next[k] += dh_raw * rnn.whh[IDX(k, i, rnn.hidden_size)];
    }
  }
}

void apply_gradients(RNN rnn, double* dwxh, double* dwhh, double* dwhy,
                     double* dbxh, double* dby, double* dh_next) {
  for (int i = 0; i < rnn.hidden_size * rnn.input_size; i++)
    rnn.wxh[i] -= lr * dwxh[i];

  for (int i = 0; i < rnn.hidden_size * rnn.hidden_size; i++)
    rnn.whh[i] -= lr * dwhh[i];

  for (int i = 0; i < rnn.output_size * rnn.hidden_size; i++)
    rnn.why[i] -= lr * dwhy[i];

  for (int i = 0; i < rnn.hidden_size; i++) rnn.bxh[i] -= lr * dbxh[i];

  for (int i = 0; i < rnn.output_size; i++) rnn.by[i] -= lr * dby[i];
}

double mean_squared_error(double* y_true, double* y_pred, int size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    double diff = y_true[i] - y_pred[i];
    sum += diff * diff;
  }
  return sum / size;
}

double mean_absolute_error(double* y_true, double* y_pred, int size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    double diff = y_true[i] - y_pred[i];
    sum += (diff < 0) ? -diff : diff;
  }
  return sum / size;
}

double regression_accuracy(double* y_true, double* y_pred, int size,
                           double tolerance) {
  int correct = 0;
  for (int i = 0; i < size; i++) {
    double diff = y_true[i] - y_pred[i];
    if ((diff < 0 ? -diff : diff) < tolerance) correct++;
  }
  return (double)correct / size;
}

int main() {
  srand(time(NULL));
  double X[] = {1, 2};
  double target[] = {2, 3};
  size_t n = sizeof(X) / sizeof(*X);

  RNN rnn = create_rnn(1, 3, 1, n);

  double* dwxh = calloc(rnn.hidden_size * rnn.input_size, sizeof(double));
  double* dwhh = calloc(rnn.hidden_size * rnn.hidden_size, sizeof(double));
  double* dwhy = calloc(rnn.output_size * rnn.hidden_size, sizeof(double));
  double* dbxh = calloc(rnn.hidden_size, sizeof(double));
  double* dby = calloc(rnn.output_size, sizeof(double));
  double* dh_next = calloc(rnn.hidden_size, sizeof(double));
  double* dloss_out = calloc(n, sizeof(double));

  // Train
  for (int e = 0; e < EPOCHS; e++) {
    memset(dwxh, 0, sizeof(double) * rnn.hidden_size * rnn.input_size);
    memset(dwhh, 0, sizeof(double) * rnn.hidden_size * rnn.hidden_size);
    memset(dwhy, 0, sizeof(double) * rnn.output_size * rnn.hidden_size);
    memset(dbxh, 0, sizeof(double) * rnn.hidden_size);
    memset(dby, 0, sizeof(double) * rnn.output_size);
    memset(dh_next, 0, sizeof(double) * rnn.hidden_size);

    forward(rnn, X, target, n);
    backward(rnn, X, target, n, dwxh, dwhh, dwhy, dbxh, dby, dh_next,
             dloss_out);
    apply_gradients(rnn, dwxh, dwhh, dwhy, dbxh, dby, dh_next);
  }

  double mse = mean_squared_error(target, rnn.y_pred, n);
  double mae = mean_absolute_error(target, rnn.y_pred, n);
  double acc = regression_accuracy(target, rnn.y_pred, n, 0.1);

  printf("MSE: %.4f | MAE: %.4f | Regression Accuracy : %.2f%%\n", mse, mae,
         acc * 100.0);

  freeRNN(rnn);
  return 0;
}
