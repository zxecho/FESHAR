from system_layer.clients.client_per import clientPer
from system_layer.servers.server_base import Server
from threading import Thread


class FedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPer)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.if_llp:
                self.LLP_receive_models()
                self.LLP_aggregate_parameters()
            else:
                self.receive_models()
                self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
