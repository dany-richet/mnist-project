from mnist.model import Model

if __name__ == "__main__":
    path_pickle = "D:\Documents\Vyve\mnist_digits_clf.pkl"
    path_img = "D:\Documents\Vyve\X_test_and_y_test.pkl"
    model = Model(path_pickle)
    res = model.predict(path_img)
    print(res)