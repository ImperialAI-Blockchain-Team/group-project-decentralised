import web3 from './web3';
import { modelDatabase} from "./abi/abi";

const address = '0x62C61ada64807521593E0845ba2592e8E83B6243';

export default new web3.eth.Contract(modelDatabase, address);