from DataPreparation import DataPreparation
from Model import Model

def main():
    model = Model('Premier League')

    while True:
        display_menu()
        u_input = input()

        if u_input == '1':
            train_model(model)
        elif u_input == '2':
            load_model(model)
        elif u_input == '3':
            load_precise_model(model)
        elif u_input == '4':
            model.display_all_model_data()
            input('Press any key to continue')
        elif u_input == '5':
            predict_match(model)
            input('\nPress any key to continue')
        elif u_input == 'e':
            break
        else:
            print("Invalid choice. Please select a valid option.")

def predict_match(model):
    print('Choose home team:\n')
    for key, value in DataPreparation.teams_as_numbers.items():
        print(f"{key}: {value}")
    print('\nChoose home team^^')
    home_team = input()
    print('\nChoose away team:\n')
    for key, value in DataPreparation.teams_as_numbers.items():
        print(f"{key}: {value}")
    print('\nChoose away team^^')
    away_team = input()
    model.predict_result(int(home_team), int(away_team))
    model.display_prediction()

def load_precise_model(model):
    model.get_model('model.pkl')
    print(f"Model loaded successfully")

def load_model(model):
    label = input(r"enter model's name: ")
    try:
        model.get_model(label)
        print(f"Model {label} loaded successfully")
    except FileNotFoundError:
        print(f"Model {label} not Found, try again")

def train_model(model):
    print(r"enter your model's name: ")
    label = input()
    print("It takes a while...")
    model.train_model(label)
    print(f"Model {label} trained successfully")

def display_menu():
    print('\nChoose option\n')
    print('Efficient model is trained already. Get it by clicking 3\n')
    print('1. Train your model')
    print('2. Load your model')
    print('3. Load the most precise model')
    print('4. Show model Parameters')
    print('5. Predict match probabilities')
    print('e. Exit')


if __name__ == "__main__":
    main()