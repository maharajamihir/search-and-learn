


        
        # Extract the hidden states from the last layer
        hidden_states = outputs.hidden_states[-1][0]
        
        # Pass the hidden states through the MLP to get logits
        logits = self.mlp(hidden_states)
        
        # Generate 1 token using the language model
        generated_token = self.lm.generate(input_ids, max_length=input_ids.size(1) + 1, num_return_sequences=1)
        
        return logits, generated_token
