pragma solidity >=0.5.16;

contract Registry {

  struct Registration {
        string user_name;
        bool data_scientist;
        bool hospital;
        bool registered;
  }

  mapping(address => Registration) public registrations;
  address[] public userHash;


// Checks if it is a User
  function isUser(address userAddress) public view returns(bool isIndeed) {
      if(registrations[userAddress].registered) {
         isIndeed = true;
      } else {
         isIndeed = false;
      }
      return (isIndeed);
  }

  // Check if user is a data scientist
  function isDataScientist(address userAddress) public view returns(bool isIndeed) {
      if(registrations[userAddress].data_scientist == true){
          isIndeed = true;
      } else {
          isIndeed = false;
      }
      return (isIndeed);
    }

  // Check is user is a data owner
  function isDataOwner(address userAddress) public view returns(bool isIndeed) {
      if(registrations[userAddress].hospital == true){
          isIndeed = true;
      } else {
          isIndeed = false;
      }
      return (isIndeed);
    }

// Inserts a User if he hasnt been registered
  function insertUser(string memory user_name, bool data_scientist, bool hospital) public returns(uint index) {
      if(isUser(msg.sender)) revert("You have already registered");
      registrations[msg.sender].registered = true;
      registrations[msg.sender].user_name = user_name;
      registrations[msg.sender].data_scientist = data_scientist;
      registrations[msg.sender].hospital = hospital;
      userHash.push(msg.sender);
      return userHash.length-1;
  }
// Retrieves the User
  function getUser(address  userAddress) public view returns(string memory user_name, bool data_scientist, bool hospital){
    if(!isUser(userAddress)) revert("This account is not registered");
    return(registrations[userAddress].user_name, registrations[userAddress].data_scientist,registrations[userAddress].hospital);
  }
// Allows to change UserType
  function updateUserType(address userAddress, bool data_scientist, bool hospital) public returns(bool success) {
    if(!isUser(userAddress)) revert("This account is not registered");
    require(userAddress == msg.sender);
    registrations[userAddress].data_scientist = data_scientist;
    registrations[userAddress].hospital = hospital;
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

   function getUsername(address userAddress) public view returns(string memory){
     return registrations[userAddress].user_name;
  }

}