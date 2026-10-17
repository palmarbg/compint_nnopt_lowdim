import torch

def truncation_selection(fitness, btm_pct=2.5):
    btm_cut = int(fitness.shape[0] * btm_pct)

    _, indices = torch.sort(fitness)
    parents = torch.cat((indices[0:-btm_cut], indices[0:btm_cut]), 0).permute((0,))

    return parents



def tournament_selection(fitness, device, group_size=2):
    population = fitness.shape[0]
    indices = torch.randint(0, population, (population, group_size), device=device)
    winners = torch.argmax(fitness[indices], 1)
    parents = indices[torch.arange(0, population, device=device), winners]
    
    return parents



def arithmetic_crossover(genomes, parents, device):
    alpha = torch.rand((genomes.shape[0], 1), device=device)
    new_genomes = torch.cat((
        alpha[0::2] * genomes[parents[0::2]] + (1 - alpha)[0::2] * genomes[parents[1::2]],
        (1 - alpha)[1::2] * genomes[parents[0::2]] + alpha[1::2] * genomes[parents[1::2]]
    ), 0)

    return new_genomes



def gaussian_mutation(genomes, device, sigma=0.1):
    return genomes + torch.randn_like(genomes, device=device) * sigma