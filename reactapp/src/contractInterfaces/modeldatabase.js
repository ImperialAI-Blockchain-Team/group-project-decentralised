import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x7cEa8Ab60435Cc9D72e43943CD9293FafD9F61f6';

export default new web3.eth.Contract(modelDatabase, address);