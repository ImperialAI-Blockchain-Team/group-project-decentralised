pragma solidity >=0.5.16;

contract Registry {

  struct Registration {
        string userName;
        bool dataScientist;
        bool hospital;
        bool registered;
  }

  mapping(address => Registration) public registrations;
  address[] public userHash;

  mapping (string => bool) public names;
  string [] public arrNames;


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
      if(registrations[userAddress].dataScientist == true){
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
  function insertUser(string memory _userName, bool _dataScientist, bool _hospital) public {
      // Check if user already registered
      if(isUser(msg.sender)){
          revert("You have already registered");
      }

      // Check if userName is unique
      if (names[_userName] == true){
          revert("User name not unique");
      }

      registrations[msg.sender].registered = true;
      registrations[msg.sender].userName = _userName;
      registrations[msg.sender].dataScientist = _dataScientist;
      registrations[msg.sender].hospital = _hospital;
      userHash.push(msg.sender);
      names[_userName] = true;
      arrNames.push(_userName);
  }
// Retrieves the User
  function getUser(address _userAddress) public view returns(string memory userName, bool dataScientist, bool hospital){

    if(!isUser(_userAddress)){
        revert("This account is not registered");
    }

    return(registrations[_userAddress].userName, registrations[_userAddress].dataScientist, registrations[_userAddress].hospital);
  }
// Allows to change UserType
  function updateUserType(address _userAddress, bool _dataScientist, bool _hospital) public {

    if(!isUser(_userAddress)) {
        revert("This account is not registered");
    }

    require(_userAddress == msg.sender,"Only user can update their account");

    registrations[_userAddress].dataScientist = _dataScientist;
    registrations[_userAddress].hospital = _hospital;

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
     return registrations[userAddress].userName;
  }

}