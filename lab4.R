install.packages("tensorflow")
library(tensorflow)
install_tensorflow()
install.packages("keras")
library(keras)
install_keras()
mnist <- dataset_mnist()
x_train <- mnist$train$x
X_test <- mnist$test$x
y_train <- mnist$train$y
y_test <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_train <- x_train / 255
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_test <- x_test / 255
y_train <- to_categorical(y_train, num_classes = 10)
y_test <- to_categorical(y_test, num_classes = 10)
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = "softmax")
summary(model)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history <- model %>%
  fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.15)
model %>%
  evaluate(x_test, y_test)
summary(model)

mnist <- dataset_mnist()
x_train <- mnist$train$x
X_test <- mnist$test$x
y_train <- mnist$train$y
y_test <- mnist$test$y
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, num_classes = 10)
y_test <- to_categorical(y_test, num_classes = 10)
model <- keras_model_sequential() %>%
   layer_flatten(input_shape = c(28, 28)) %>%
   layer_dense(units = 128, activation = "relu") %>%
   layer_dense(units = 10, activation = "softmax")
summary(model)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history <- model %>%
   fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.15)
model %>%
  evaluate(x_test, y_test)
plot(history)