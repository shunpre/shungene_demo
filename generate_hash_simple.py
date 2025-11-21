import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher.hash_list(['123'])
print(hashed_passwords[0])
