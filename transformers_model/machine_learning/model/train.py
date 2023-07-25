import numpy as np
import tensorflow as tf

def train_model_with_seed(seed, X_train, y_train, X_test,y_test, epochs=50,lt_epochs = 20, batch_size=64,d_model=128,num_heads=5,num_layers=3,lr=0.002):
    # Set random seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create the model

    model = create_transformer_model(
        input_shape=(seq_len, X_train.shape[-1]), d_model=d_model, num_heads=num_heads, num_layers=num_layers
    )

    # Define the model checkpoint callback
    checkpoint_filepath = 'best_model_checkpoint.h5'
    test_best_checkpoint = TestBestModelCheckpoint(test_data=(X_test, y_test),
                                                  metric='loss',
                                                  mode='min',
                                                  save_path=checkpoint_filepath)


    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")
    #model.load_weights('best_model.h5')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), validation_split=0.1,callbacks = [test_best_checkpoint])
    # Define and compile the model
    model = create_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]), d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    model.compile(optimizer="adam", loss="mae")

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    model.load_weights(checkpoint_filepath)

    best_epoch = 0
    best_error = float('inf')
    best_weights = None
    n_past = 30

    for epoch in range(lt_epochs):
        print(f"Epoch {epoch + 1}/{lt_epochs}")
        
        # Train the model for one epoch
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test))
        
        # Evaluate the long-term forecast
        error = evaluate_forecasts(model, stock_data_preprocessed, seq_len, n_past, scaler)
        error2 = evaluate_forecasts(model, stock_data_preprocessed[:len(stock_data_preprocessed)-n_past], seq_len, n_past, scaler)
        error3 = evaluate_forecasts(model, stock_data_preprocessed[:len(stock_data_preprocessed)-(n_past*2)], seq_len, n_past, scaler)

        print(f"Long-term forecast error: {error+error2+error3}")
        
        # Save the model weights if the error is the lowest so far
        if (error + error2 + error3) < best_error:
            best_epoch = epoch
            best_error = error + error2 + error3
            best_weights = model.get_weights()
            print("Best model found so far.")

    print(f"Best model found at epoch {best_epoch + 1} with long-term forecast error: {best_error}")

    # Load the best weights into the model
    model.set_weights(best_weights)

    return model

def ensemble_prediction(models, input_data):
    predictions = []

    for model in models:
        prediction = model.predict(input_data)
        predictions.append(prediction)

    # Average the predictions
    ensemble_prediction = np.mean(predictions, axis=0)

    return ensemble_prediction