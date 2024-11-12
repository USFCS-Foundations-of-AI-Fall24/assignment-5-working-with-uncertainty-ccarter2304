from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD
def get_queries() :
    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery":['Works',"Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas":['Full',"Empty"]},
    )

    cpd_keyPresent = TabularCPD(
        variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
        state_names={"KeyPresent": ['yes', 'no']}
    )

    cpd_radio = TabularCPD(
        variable=  "Radio", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                     "Battery": ['Works',"Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable=  "Ignition", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                     "Battery": ['Works',"Doesn't work"]}
    )

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent": ['yes', 'no']},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01],[0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                     "Starts": ['yes', 'no'] }
    )



    # Associating the parameters with the model structure
    car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keyPresent)

    car_infer = VariableElimination(car_model)

    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))
    print("Query 1: Probability that battery is not working given car will not move")
    print (car_infer.query(variables=["Battery"], evidence={"Moves":"no"}))
    print("Query 2: Probability that car will not start given radio is not working")
    print(car_infer.query(variables=["Moves"], evidence={"Radio":"Doesn't turn on"}))
    print("Query 3-1: Probability of radio working before discovering gas in car")
    print( car_infer.query(variables=["Radio"], evidence={"Battery":"Works"}))
    print("Query 3-2: Probability of radio working after discovering gas in car")
    print( car_infer.query(variables=["Radio"], evidence={"Battery":"Works", "Gas": "Full"}))

    # Given that the car doesn't move, how does the probability of the ignition failing change
    # if we observe that the car dies not have gas in it?
    print("Query 4-1: Probability of the ignition failing given that the car does not move before observation of gas")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves":"no"}) )#, "Gas": "full"})
    print("Query 4-2: Probability of the ignition failing given that the car does not move after observation of gas")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves":"no", "Gas": "Empty"}))

    print("Query 5: Probability that the car starts if the radio works and it has gas in it")
    print(car_infer.query(variables=["Starts"], evidence={"Radio":"turns on", "Gas": "Full"}))
    print("Query 6: Probability that the key is not present given that the car does not move")
    print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))

if __name__ == '__main__' :
    get_queries()





