import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xa91A90BB6C18E60cAf41Db14A818687b5845d66a';

export default new web3.eth.Contract(modelDatabase, address);