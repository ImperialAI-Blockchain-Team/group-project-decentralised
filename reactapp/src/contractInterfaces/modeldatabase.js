import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x93ED36F5504E3aAc18e739Bfcd4d0a19afeb3ce6';

export default new web3.eth.Contract(modelDatabase, address);