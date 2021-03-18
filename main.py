# Evaluate model that has already been trained


'''
gen_to_eval = Generator(z_dim, hidden_dim_g, num_layers).to(device)
gen_to_eval.load_state_dict(torch.load(f'pretrained_models/{model_name}'))
gen_to_eval.eval()
'''