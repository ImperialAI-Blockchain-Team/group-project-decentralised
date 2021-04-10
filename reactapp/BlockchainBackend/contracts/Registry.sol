// SPDX-License-Identifier: MIT
pragma solidity >=0.5.16;


contract Registry {

  struct Registration {
        string user_name;
        bool data_scientist;
        bool aggregator;
        bool hospital;
        bool registered;
  }

  mapping(address => Registration) private registrations;
  address[] private userHash;

  event LogNewUser (address indexed userAddress,  string user_name, bool data_scientist, bool aggregator, bool hospital);
  event LogUpdateUser(address indexed userAddress, string user_name, bool data_scientist, bool aggregator, bool hospital);

// Checks if it is a User
  function isUser(address userAddress) public view returns(bool isIndeed) {
      if(registrations[userAddress].registered) {
         isIndeed = true;
      } else {
         isIndeed = false;
      }
      return (isIndeed);
  }
// Inserts a User if he hasnt been registered
  function insertUser(string memory user_name, bool data_scientist, bool aggregator, bool hospital) public returns(uint index) {
      if(isUser(msg.sender)) revert("You have already registered");
      registrations[msg.sender].registered = true;
      registrations[msg.sender].user_name = user_name;
      registrations[msg.sender].data_scientist = data_scientist;
      registrations[msg.sender].aggregator = aggregator;
      registrations[msg.sender].hospital = hospital;
      userHash.push(msg.sender);
      emit LogNewUser(msg.sender, user_name, data_scientist,aggregator,hospital);
      return userHash.length-1;
  }
// Retreives the User
  function getUser(address  userAddress) public view returns(string memory user_name, bool data_scientist, bool aggregator, bool hospital){
    if(!isUser(userAddress)) revert("This account is not registered");
    return(registrations[userAddress].user_name, registrations[userAddress].data_scientist,registrations[userAddress].aggregator,registrations[userAddress].hospital);
  }
// Allows to change UserType
  function updateUserType(address userAddress, bool data_scientist, bool aggregator, bool hospital) public returns(bool success) {
    if(!isUser(userAddress)) revert("This account is not registered");
    require(userAddress == msg.sender);
    registrations[userAddress].data_scientist = data_scientist;
    registrations[userAddress].aggregator = aggregator;
    registrations[userAddress].hospital = hospital;
    emit LogUpdateUser(userAddress, registrations[userAddress].user_name, data_scientist,aggregator,hospital);
    return true;
  }
// Returns count of Registered Users
  function getUserCount() public view returns(uint count) {
    return userHash.length;
  }

  function getUserAtIndex(uint index) public view returns(address userAddress) {
    return userHash[index];
  }

   function deleteUser(address userAddress) public {
    require(userAddress == msg.sender);
    delete(registrations[userAddress]);
  }

}