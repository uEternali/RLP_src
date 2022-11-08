from utils.loggerinfo import get_logger
from utils.earlystopping import EarlyStopping
from datamaker2 import generate_dataloader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from model2 import RLPnet
import numpy as np
import torch
from torch import optim
from torch import nn
from utils.loggerinfo import get_logger


class Engine():
    def __init__(self, args, loggername='LSTM'):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger('LSTM.txt', loggername)
        self.model = RLPnet(len(self.args.catelist), self.args.predict_seq_window, self.args.attn_src_len, self.args.season_seq_window!=0, self.args.trend_seq_window!=0)
        self.model.to(self.device)
    
    def train_one_epoch(self, train_loader, optimizer, criterion):
        model = self.model.train()
        train_loss, itercnt = 0.0, 0
        for _,  (batch_x, batch_y) in enumerate(train_loader):
            x_seq = [xi.to(self.device).float() for xi in batch_x[0]]
            x_stamp = [xi.to(self.device).float() for xi in batch_x[1]]
            y = batch_y.to(self.device).float()
            single_batch_size = y.shape[0]

            y_pred = model(x_seq, x_stamp)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss * single_batch_size
            itercnt += single_batch_size
        #     print(">>> Training [{}/{} ({:.2f}%)] MSELoss:{:.6f}".format(
        #         itercnt, len(train_loader.dataset), 100.0 * itercnt / len(train_loader.dataset), loss.item()), end="\r"
        #     )
        # print("")
        train_loss /= len(train_loader.dataset)
        return train_loss.item()


    def validate_one_epoch(self, valid_loader, criterion):
        model = self.model.eval()
        valid_loss, itercnt = 0.0, 0
        with torch.no_grad():
            for _,  (batch_x, batch_y) in enumerate(valid_loader):
                x_seq = [xi.to(self.device).float() for xi in batch_x[0]]
                x_stamp = [xi.to(self.device).float() for xi in batch_x[1]]
                y = batch_y.to(self.device).float()
                single_batch_size = y.shape[0]

                y_pred = model(x_seq, x_stamp)

                loss = criterion(y_pred, y)

                valid_loss += loss * single_batch_size
                itercnt += single_batch_size
            #     print(">>> Validating [{}/{} ({:.2f}%)] MSELoss:{:.6f}".format(
            #         itercnt, len(valid_loader.dataset), 100.0 * itercnt / len(valid_loader.dataset), loss.item()), end="\r"
            #     )
            # print("")
            valid_loss /= len(valid_loader.dataset)
        return valid_loss.item()

    def test(self, test_loader, load=True):
        model = self.model.eval()
        if load: model.load_state_dict(torch.load("checkpoint.pth"))
        prediction, groundtruth, itercnt = [], [], 0
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(test_loader):
                x_seq = [xi.to(self.device).float() for xi in batch_x[0]]
                x_stamp = [xi.to(self.device).float() for xi in batch_x[1]]
                y = batch_y.to(self.device).float()
                single_batch_size = y.shape[0]

                y_pred = model(x_seq, x_stamp).cpu().data.numpy()

                y_true = y.cpu().data.numpy()

                inversed_y_pred = test_loader.dataset.inverse_transform(y_pred[0].reshape(1, -1))
                inversed_y_true = test_loader.dataset.inverse_transform(y_true[0].reshape(1, -1))

                prediction.append(inversed_y_pred.squeeze(0))
                groundtruth.append(inversed_y_true.squeeze(0))

                itercnt += single_batch_size
                # print(">>> Testing [{}/{} ({:.2f}%)]".format(
                #     itercnt, len(test_loader.dataset),
                #     100.0 * itercnt / len(test_loader.dataset)), end="\r"
                # )

        rmse_error = np.sqrt(mean_squared_error(np.asarray(prediction), np.asarray(groundtruth)))
        mea_error = mean_absolute_error(np.asarray(prediction), np.asarray(groundtruth))
       
        self.logger.info('ALL RMSE: \t\t{:.6f}'.format(rmse_error.item()))
        self.logger.info('ALL MEA: \t\t{:.6f}'.format(mea_error.item()))

        return rmse_error, mea_error

    def train(self):
        train_loader = generate_dataloader(self.args, 'train')
        valid_loader = generate_dataloader(self.args, 'valid')
        test_loader = generate_dataloader(self.args, 'test')
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = nn.MSELoss()

        epochs = self.args.train_epochs
        for epoch in range(epochs):
            train_losses, valid_losses = [], []
            train_loss = self.train_one_epoch(train_loader, optimizer, criterion)
            valid_loss = self.validate_one_epoch(valid_loader, criterion)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            self.logger.info("Train Epoch[{}/{}]: Average Train MSELoss: {:.6f} | Average Validate MSELoss: {:.6f}".format(
                epoch, epochs, train_loss, valid_loss))

            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.logger.info("Early Stopped!")
                break
            if self.args.predict_seq_window != 1:
                self.test(test_loader, load=False)

        rmse, mae = self.test(test_loader)
        return rmse, mae