import argparse
import matplotlib.pyplot as plt
import json

def plot_from_json(filename):
    with open(filename, "r") as f:
        json_data = json.load(f)
        
        client_num, experiment, max_round = json_data["client number"], json_data["experiment"], json_data["max round"]

        client_acc, server_acc = json_data["clients acc"], json_data["server acc"]

        plt.figure(1, figsize=(12, 12))
        plt.title("Client num: "+str(client_num)+" experiment: "+str(experiment)+" max round: "+str(max_round))
        plt.xlabel("rounds")
        plt.ylabel("accuracy (%)") 
        plt.xticks([i for i in range(max_round)])
        
        for client_id, accs in client_acc.items():
            plt.plot(accs, label="client " + str(client_id))
        
        plt.plot(server_acc, label="server accuracy")
        plt.legend()
        
        pltfilename = filename.replace("json", "png")
        plt.savefig(pltfilename)
        print("plot saved in", pltfilename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-n", required=True, type=str)
    args = parser.parse_args()
    plot_from_json(args.filename)


