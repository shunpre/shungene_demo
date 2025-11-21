import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher.hash_list(['admin'])
print(hashed_passwords[0])
