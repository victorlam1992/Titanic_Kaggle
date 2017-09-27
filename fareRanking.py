def top_rich(dataset, amount, cat, exclude):
    df_rank = dataset
    if cat == 'rich':
        if exclude == True:
            df_rank = df_rank[df_rank['Pclass'] != 3]
        top_rich = df_rank.nlargest(amount, 'Fare')[['Pclass', 'Fare', 'Survived']]
        print("Survival rate for top %i richest: %f" % (amount, top_rich['Survived'].mean()))
        return top_rich
        
    elif cat == 'poor':
        if exclude == True:
            df_rank = df_rank[df_rank['Pclass'] != 1]
        top_poor = df_rank.nsmallest(amount, 'Fare')[['Pclass', 'Fare', 'Survived']]
        print("Survival rate for bottom %i richest: %f" % (amount, top_poor['Survived'].mean()))
        return top_poor