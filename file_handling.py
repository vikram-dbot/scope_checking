import pandas as pd

#loading and cleaning thr data
ai_data = pd.read_csv('D:/AI data .csv')
construction_data = pd.read_csv('C:/Users/vikra/java-class/construction_progress_results.csv')
ai_data = ai_data.fillna(0)
construction_data.iloc[:,1:] = construction_data.iloc[:,1:].map(lambda x: 1 if int(x) > 0 else 0)

#calculate accuracy of the individual columns
def accuracy_calculation(name):
    count = 0
    for i in range(len(construction_data[name])):
        if construction_data[name][i] == ai_data[name][i]:
            count += 1
    accuracy = count / len(construction_data[name]) * 100
    print(f"Accuracy of {name}: {accuracy:.2f}%")
    print()
    return accuracy

#calculation of accuracy for block work 
block_work_yet_to_start_accuracy = accuracy_calculation('block_work_yet_to_start')
block_work_in_progress_accuracy = accuracy_calculation('block_work_in_progress')
block_work_completed_accuracy = accuracy_calculation('block_work_completed')

#calculation of accuracy for plastering
plastering_yet_to_start_accuracy = accuracy_calculation('cement_plastering_yet_to_start')
plastering_in_progress_accuracy = accuracy_calculation('cement_plastering_in_progress')
plastering_completed_accuracy = accuracy_calculation('cement_plastering_completed')

#calculation of accuracy for flooring
flooring_yet_to_start_accuracy = accuracy_calculation('flooring_yet_to_start')
flooring_in_progress_accuracy = accuracy_calculation('flooring_in_progress')
flooring_completed_accuracy = accuracy_calculation('flooring_completed')

#calculation of accuracy for windows and ventillation installation
windows_and_ventillation_yet_to_start_accuracy = accuracy_calculation('windows_&_ventilators_fixing_yet_to_start')
windows_and_ventillation_in_progress_accuracy = accuracy_calculation('windows_&_ventilators_fixing_in_progress')
windows_and_ventillation_completed_accuracy = accuracy_calculation('windows_&_ventilators_fixing_completed')

#calculation of accuracy for electrical_switches_&_sockets_installation
electrical_switches_yet_to_start_accuracy = accuracy_calculation('electrical_switches_&_sockets_installation_yet_to_start')
electrical_switches_in_progress_accuracy = accuracy_calculation('electrical_switches_&_sockets_installation_in_progress')
electrical_switches_completed_accuracy = accuracy_calculation('electrical_switches_&_sockets_installation_completed')

#printing the accuracy
'''
print(f"Accuracy for block work: {(block_work_completed_accuracy+block_work_in_progress_accuracy+block_work_yet_to_start_accuracy)/3:.2f}%")
print()

print(f"Accuracy for cement plastering: {(plastering_yet_to_start_accuracy+plastering_in_progress_accuracy+plastering_completed_accuracy)/3:.2f}%")
print()

print(f"Accuracy for flooring: {(flooring_yet_to_start_accuracy+flooring_in_progress_accuracy+flooring_completed_accuracy)/3:.2f}%")
print()

print(f"Accuracy for windows and ventillation: {(windows_and_ventillation_yet_to_start_accuracy+windows_and_ventillation_in_progress_accuracy+windows_and_ventillation_completed_accuracy)/3:.2f}%")
print()

print(f"Accuracy for electrical switches: {(electrical_switches_yet_to_start_accuracy+electrical_switches_in_progress_accuracy+electrical_switches_completed_accuracy)/3:.2f}")
print()
'''