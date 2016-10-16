from utilities import *

format = '%i'

data_folder = './data'

filepath0 = './CrisisLexT6/2012_Sandy_Hurricane/2012_Sandy_Hurricane_0_final.txt'
filepath1 = './CrisisLexT6/2012_Sandy_Hurricane/2012_Sandy_Hurricane_1_final.txt'

sandyData, sandyLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/sandyData.csv', sandyData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/sandyLabels.csv', sandyLabels, fmt=format, delimiter=',')
#-------------------------------------------------------------------------------

filepath0 = './CrisisLexT6/2013_Alberta_Floods/2013_Alberta_Floods_0_final.txt'
filepath1 = './CrisisLexT6/2013_Alberta_Floods/2013_Alberta_Floods_1_final.txt'

albertaData, albertaLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/albertaData.csv', albertaData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/albertaLabels.csv', albertaLabels, fmt=format, delimiter=',')
#--------------------------------------------------------------------------------

filepath0 = './CrisisLexT6/2013_Boston_Bombings/2013_Boston_Bombings_0_final.txt'
filepath1 = './CrisisLexT6/2013_Boston_Bombings/2013_Boston_Bombings_1_final.txt'

bostonData, bostonLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/bostonData.csv', bostonData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/bostonLabels.csv', bostonLabels, fmt=format, delimiter=',')
#--------------------------------------------------------------------------------

filepath0 = './CrisisLexT6/2013_Oklahoma_Tornado/2013_Oklahoma_Tornado_0_final.txt'
filepath1 = './CrisisLexT6/2013_Oklahoma_Tornado/2013_Oklahoma_Tornado_1_final.txt'

oklahomaData, oklahomaLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/oklahomaData.csv', oklahomaData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/oklahomaLabels.csv', oklahomaLabels, fmt=format, delimiter=',')
#--------------------------------------------------------------------------------

filepath0 = './CrisisLexT6/2013_Queensland_Floods/2013_Queensland_Floods_0_final.txt'
filepath1 = './CrisisLexT6/2013_Queensland_Floods/2013_Queensland_Floods_1_final.txt'

queenslandData, queenslandLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/queenslandData.csv', queenslandData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/queenslandLabels.csv', queenslandLabels, fmt=format, delimiter=',')
#--------------------------------------------------------------------------------

filepath0 = './CrisisLexT6/2013_West_Texas_Explosion/2013_West_Texas_Explosion_0_final.txt'
filepath1 = './CrisisLexT6/2013_West_Texas_Explosion/2013_West_Texas_Explosion_1_final.txt'

westtexasData, westtexasLabels = get_clean_data_clean_labels(filepath0, filepath1)
np.savetxt(data_folder+'/westtexasData.csv', westtexasData, fmt=format, delimiter=',')
np.savetxt(data_folder+'/westtexasLabels.csv', westtexasLabels, fmt=format, delimiter=',')